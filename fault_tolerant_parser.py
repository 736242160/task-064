#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fault_tolerant_parser.py — 容错记录流解析器（纯 Python 标准库，单文件）

记录流格式
----------
每条记录以标记行开头、以 @END 结尾，中间是 1..n 行 CSV 数据：

    @BEGIN <record_id> <field_count>
    field1,field2,...,fieldN
    ...
    @END

- @BEGIN 声明记录 id 与每行应有的字段数；
- 数据行是单行 CSV（字段可带引号，引号内可含逗号）；
- 记录之外的内容（空行、杂散文本）被忽略。

恢复策略：跳过整条坏记录（panic-mode），不尝试修复
-------------------------------------------------
1. 标记行 @BEGIN/@END 是可靠的“再同步点”，跳过的代价只是丢一条记录；
2. 修复（补引号、截断/填充字段）需要猜测数据语义，可能把静默错误引入
   下游且不可审计；跳过是确定性的，坏记录原样留在输入中可供人工核查；
3. 判坏后进入 skip 模式：丢弃该记录已累积的全部状态（已解析行、期望
   字段数），消费输入直到 @END；若先遇到下一条 @BEGIN，补报 missing-end
   后正常开始新记录。

防状态泄漏的关键设计
--------------------
- CSV 解析器按行新建（csv.reader 不跨行复用），引号状态绝不跨行、跨记录；
- 每条记录的全部状态（id、起始行、期望字段数、已解析行）保存在局部变量，
  记录结束（成功或失败）即整体销毁，下一条记录从零开始；
- 行分类器 _tokenize 是无状态纯函数。

错误类别（kind）
----------------
- unclosed-quote        数据行引号未闭合
- field-count-mismatch  数据行字段数与 @BEGIN 声明不符
- csv-error             其他 CSV 语法错误（如引号后未跟分隔符）
- bad-marker            @BEGIN 行格式非法（缺参数 / 字段数不是正整数）
- missing-end           记录缺少 @END（遇到下一条 @BEGIN 或 EOF）
- stray-end             记录之外出现孤立的 @END
- empty-record          记录没有任何数据行

同一次运行报告全部坏记录：一条记录判坏只影响它自己，解析器恢复后继续。

用法
----
    python3 fault_tolerant_parser.py 输入文件          # 解析文件
    python3 fault_tolerant_parser.py < 输入文件        # 从 stdin 读
    python3 fault_tolerant_parser.py --json 输入文件   # 机器可读输出
    python3 fault_tolerant_parser.py --demo            # 运行内置示例
    python3 fault_tolerant_parser.py --selftest        # 运行自检

退出码：0 = 无错误；1 = 存在坏记录；2 = 用法/IO 错误。

输入示例（即 --demo 内置样例）
------------------------------
    @BEGIN r1 3
    alpha,beta,gamma
    1,2,3
    @END
    @BEGIN r2 2
    "unclosed,quote
    x,y
    @END
    @BEGIN r3 2
    ok1,ok2
    @END

输出示例
--------
    === Records (2) ===
    [r1] starts at line 1, 2 row(s)
        ['alpha', 'beta', 'gamma']
        ['1', '2', '3']
    [r3] starts at line 9, 1 row(s)
        ['ok1', 'ok2']
    === Errors (1) ===
    line 5 [r2] unclosed-quote: line 6: unexpected end of data
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import asdict, dataclass


@dataclass
class Record:
    record_id: str
    start_line: int
    rows: list  # list[list[str]]


@dataclass
class RecordError:
    line: int       # 记录起始行（标记类错误则为标记所在行）
    record_id: str
    kind: str
    detail: str

    def __str__(self) -> str:
        return f"line {self.line} [{self.record_id}] {self.kind}: {self.detail}"


def _tokenize(line, lineno):
    """把一行文本分类为事件。无状态纯函数，本身不可能泄漏解析状态。"""
    if line.strip() == "@END":
        return ("end", lineno, None)
    if line.startswith("@BEGIN"):
        parts = line.split()
        if len(parts) != 3:
            return ("badbegin", lineno,
                    f"expected '@BEGIN <id> <field_count>', got {line!r}")
        try:
            nfields = int(parts[2])
        except ValueError:
            return ("badbegin", lineno,
                    f"field_count is not an integer: {parts[2]!r}")
        if nfields <= 0:
            return ("badbegin", lineno,
                    f"field_count must be positive, got {nfields}")
        return ("begin", lineno, (parts[1], nfields))
    return ("data", lineno, line)


def _parse_csv_row(line):
    """解析单行 CSV。每次调用新建 reader —— 引号状态不跨行，这是防泄漏的关键。"""
    return next(csv.reader([line], strict=True))


class RecordStreamParser:
    EXPECT, IN_REC, SKIP = "expect-begin", "in-record", "skip-bad-record"

    def __init__(self):
        self.records = []
        self.errors = []

    def parse(self, stream):
        state = self.EXPECT
        # 当前记录的全部状态都收在这四个局部变量里，记录结束即整体失效
        rec_id = rec_start = nfields = rows = None

        def fail(kind, detail):
            self.errors.append(RecordError(rec_start, rec_id, kind, detail))

        for lineno, raw in enumerate(stream, 1):
            line = raw.rstrip("\r\n")
            event, _, payload = _tokenize(line, lineno)

            if state == self.EXPECT:
                if event == "begin":
                    rec_id, nfields = payload
                    rec_start, rows = lineno, []
                    state = self.IN_REC
                elif event == "badbegin":
                    self.errors.append(RecordError(lineno, "?", "bad-marker", payload))
                elif event == "end":
                    self.errors.append(RecordError(
                        lineno, "-", "stray-end", "@END outside any record"))
                # data：记录之外的内容，忽略

            elif state == self.IN_REC:
                if event in ("data", "badbegin"):
                    # 记录内的非法 @BEGIN 不当成新记录，按普通数据处理
                    text = payload if event == "data" else line
                    try:
                        row = _parse_csv_row(text)
                    except csv.Error as exc:
                        msg = str(exc)
                        kind = ("unclosed-quote"
                                if "unexpected end of data" in msg else "csv-error")
                        fail(kind, f"line {lineno}: {msg}")
                        state = self.SKIP
                        continue
                    if len(row) != nfields:
                        fail("field-count-mismatch",
                             f"line {lineno}: expected {nfields} field(s), "
                             f"got {len(row)}")
                        state = self.SKIP
                        continue
                    rows.append(row)
                elif event == "end":
                    if rows:
                        self.records.append(Record(rec_id, rec_start, rows))
                    else:
                        fail("empty-record", "record has no data rows")
                    state = self.EXPECT
                elif event == "begin":
                    fail("missing-end",
                         f"record not closed: hit new @BEGIN at line {lineno}")
                    rec_id, nfields = payload
                    rec_start, rows = lineno, []
                    # 保持 IN_REC，新记录从此开始

            else:  # SKIP：丢弃坏记录残余，寻找再同步点
                if event == "end":
                    state = self.EXPECT
                elif event == "begin":
                    fail("missing-end",
                         f"bad record not closed: hit new @BEGIN at line {lineno}")
                    rec_id, nfields = payload
                    rec_start, rows = lineno, []
                    state = self.IN_REC
                # data/badbegin：坏记录的残留，丢弃

        if state == self.IN_REC:
            fail("missing-end", "record not closed: reached EOF")
        elif state == self.SKIP:
            fail("missing-end", "bad record not closed: reached EOF while skipping")
        return self.records, self.errors


def render_text(records, errors, out):
    out.write(f"=== Records ({len(records)}) ===\n")
    for rec in records:
        out.write(f"[{rec.record_id}] starts at line {rec.start_line}, "
                  f"{len(rec.rows)} row(s)\n")
        for row in rec.rows:
            out.write(f"    {row!r}\n")
    out.write(f"=== Errors ({len(errors)}) ===\n")
    for err in errors:
        out.write(f"  {err}\n")


def render_json(records, errors, out):
    json.dump({
        "records": [asdict(r) for r in records],
        "errors": [asdict(e) for e in errors],
    }, out, ensure_ascii=False, indent=2)
    out.write("\n")


DEMO_INPUT = """\
@BEGIN r1 3
alpha,beta,gamma
1,2,3
@END
@BEGIN r2 2
"unclosed,quote
x,y
@END
@BEGIN r3 2
ok1,ok2
@END
@BEGIN r4 3
only,two
@END
@END
@BEGIN broken
@BEGIN r5 1
solo
@BEGIN r6 2
tail1,tail2
"""


def _selftest():
    def parse_text(text):
        return RecordStreamParser().parse(io.StringIO(text))

    # 1. 正常记录
    recs, errs = parse_text("@BEGIN a 2\nx,y\n@END\n")
    assert len(recs) == 1 and not errs and recs[0].rows == [["x", "y"]]

    # 2. 引号未闭合 → unclosed-quote，且后续记录完整解析（无状态泄漏）
    recs, errs = parse_text('@BEGIN a 1\n"oops\n@END\n@BEGIN b 2\nx,y\n@END\n')
    assert [r.record_id for r in recs] == ["b"]
    assert recs[0].rows == [["x", "y"]]
    assert errs[0].kind == "unclosed-quote" and errs[0].line == 1

    # 3. 字段数不符 → field-count-mismatch，与引号错误区分开
    recs, errs = parse_text("@BEGIN a 3\n1,2\n@END\n@BEGIN b 1\nz\n@END\n")
    assert [r.record_id for r in recs] == ["b"]
    assert errs[0].kind == "field-count-mismatch"

    # 4. 缺 @END：遇到下一条 @BEGIN 时报 missing-end，新记录正常解析
    recs, errs = parse_text("@BEGIN a 1\nx\n@BEGIN b 1\ny\n@END\n")
    assert [r.record_id for r in recs] == ["b"]
    assert errs[0].kind == "missing-end"

    # 5. 坏记录缺 @END：skip 中遇到 @BEGIN，补报 missing-end 并恢复
    recs, errs = parse_text('@BEGIN a 1\n"bad\n@BEGIN b 1\ny\n@END\n')
    assert [r.record_id for r in recs] == ["b"]
    assert [e.kind for e in errs] == ["unclosed-quote", "missing-end"]

    # 6. 一次运行报告全部坏记录，不停在第一个
    recs, errs = parse_text(
        "@BEGIN a 2\n1\n@END\n@BEGIN b 2\n1,2,3\n@END\n@BEGIN c 1\nok\n@END\n")
    assert [r.record_id for r in recs] == ["c"]
    assert [e.record_id for e in errs] == ["a", "b"]

    # 7. 孤立 @END / 非法标记 / 空记录
    _, errs = parse_text("@END\n@BEGIN x\n@BEGIN e 2\n@END\n")
    assert [e.kind for e in errs] == ["stray-end", "bad-marker", "empty-record"]

    # 8. EOF 时记录未闭合
    _, errs = parse_text("@BEGIN a 1\nx\n")
    assert errs[-1].kind == "missing-end"

    print("selftest: all 8 checks passed")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="容错记录流解析器（格式与策略见模块 docstring）")
    ap.add_argument("input", nargs="?", default="-",
                    help="输入文件路径，缺省或 '-' 表示标准输入")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    ap.add_argument("--demo", action="store_true", help="运行内置示例")
    ap.add_argument("--selftest", action="store_true", help="运行自检")
    args = ap.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if args.demo:
        sys.stdout.write("----- input -----\n" + DEMO_INPUT
                         + "----- output -----\n")
        records, errors = RecordStreamParser().parse(io.StringIO(DEMO_INPUT))
    else:
        try:
            if args.input == "-":
                records, errors = RecordStreamParser().parse(sys.stdin)
            else:
                with open(args.input, "r", encoding="utf-8", newline="") as fh:
                    records, errors = RecordStreamParser().parse(fh)
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.json:
        render_json(records, errors, sys.stdout)
    else:
        render_text(records, errors, sys.stdout)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
