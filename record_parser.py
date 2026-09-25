#!/usr/bin/env python3
"""
容错多行记录解析器。

记录格式：
- 标记行必须是独立的一行：##RECORD
- 每个字段占一个物理行，格式为 key=value
- 必需字段恰好为 id、name、email 三个
- value 可以用英文单引号或双引号括起来
- 引号中的 \n、\t、\r、\\、\"、\' 会被转义
- 引号可以跨行；直到引号关闭前，后续物理行都是字段内容
- 标记行是硬同步点；即使引号未关闭，遇到下一个 ##RECORD 也开始新记录

用法：
    python3 record_parser.py --demo
    python3 record_parser.py records.txt
    cat records.txt | python3 record_parser.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


MARKER = "##RECORD"
REQUIRED_FIELDS = ("id", "name", "email")
ALLOWED_FIELDS = frozenset(REQUIRED_FIELDS)


@dataclass
class Reason:
    code: str
    message: str
    line: Optional[int] = None
    field: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "code": self.code,
            "reason": self.message,
        }
        if self.line is not None:
            data["line"] = self.line
        if self.field is not None:
            data["field"] = self.field
        return data


@dataclass
class BadRecord:
    start_line: int
    end_line: int
    reasons: List[Reason]

    def to_dict(self) -> Dict[str, object]:
        return {
            "start_line": self.start_line,
            "end_line": self.end_line,
            "primary_code": self.reasons[0].code,
            "reasons": [reason.to_dict() for reason in self.reasons],
        }


@dataclass
class ParseResult:
    records: List[Dict[str, str]]
    bad_records: List[BadRecord]
    stream_errors: List[Reason]

    def to_dict(self) -> Dict[str, object]:
        return {
            "records": self.records,
            "errors": {
                "bad_records": [item.to_dict() for item in self.bad_records],
                "stream_errors": [item.to_dict() for item in self.stream_errors],
            },
        }


class RecordState:
    def __init__(self, start_line: int) -> None:
        self.start_line = start_line
        self.end_line = start_line
        self.fields: Dict[str, str] = {}
        self.field_lines: Dict[str, int] = {}
        self.field_count = 0
        self.reasons: List[Reason] = []
        self.fatal = False
        self.open_key: Optional[str] = None
        self.open_line: Optional[int] = None
        self.open_quote: Optional[str] = None
        self.quote_buffer: List[str] = []

    def feed(self, line: str, line_number: int) -> None:
        self.end_line = line_number
        if self.fatal:
            return
        try:
            if self.open_key is not None:
                self._feed_quoted_continuation(line, line_number)
            elif not line.strip():
                return
            else:
                self._feed_field_line(line, line_number)
        except Exception as exc:
            self.fatal = True
            self._reset_quote()
            self.reasons.append(
                Reason(
                    "internal_parse_error",
                    f"记录解析器发生内部错误，已丢弃整条记录：{exc}",
                    line_number,
                )
            )

    def finish(self) -> Tuple[Optional[Dict[str, str]], Optional[BadRecord], bool]:
        if not self.fatal and self.open_key is not None:
            self.reasons.append(
                Reason(
                    "unclosed_quote",
                    f"字段 {self.open_key} 的引号未闭合",
                    self.open_line,
                    self.open_key,
                )
            )
            self._reset_quote()

        if not self.fatal:
            self._validate_fields()

        self.reasons.sort(
            key=lambda reason: (
                reason.line if reason.line is not None else self.start_line,
                reason.code,
                reason.message,
            )
        )

        if self.fatal or self.reasons:
            return None, BadRecord(self.start_line, self.end_line, self.reasons), self.fatal

        record = {name: self.fields[name] for name in REQUIRED_FIELDS}
        return record, None, False

    def _feed_field_line(self, line: str, line_number: int) -> None:
        if "=" not in line:
            self.reasons.append(
                Reason(
                    "malformed_field",
                    "字段行必须包含等号，格式为 key=value",
                    line_number,
                )
            )
            return

        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key or not all(ch.isalnum() or ch == "_" for ch in key):
            self.reasons.append(
                Reason(
                    "malformed_field",
                    "字段名必须非空，且只能包含字母、数字和下划线",
                    line_number,
                )
            )
            return

        normalized_key = key.lower()
        value = raw_value.lstrip()

        if value[:1] in ("'", '"'):
            self.open_key = normalized_key
            self.open_line = line_number
            self.open_quote = value[0]
            self.quote_buffer = []
            closed = self._consume_quoted(value, 1, line_number, normalized_key)
            if closed:
                self._store_open_field(line_number)
        else:
            self._store_field(normalized_key, raw_value.strip(), line_number)

    def _feed_quoted_continuation(self, line: str, line_number: int) -> None:
        self.quote_buffer.append("\n")
        closed = self._consume_quoted(line, 0, line_number, self.open_key or "")
        if closed:
            self._store_open_field(line_number)

    def _consume_quoted(
        self,
        text: str,
        start: int,
        line_number: int,
        key: str,
    ) -> bool:
        quote = self.open_quote
        position = start
        segment: List[str] = []

        while position < len(text):
            char = text[position]
            if char == "\\":
                if position + 1 >= len(text):
                    self.reasons.append(
                        Reason(
                            "invalid_escape",
                            "行尾反斜杠后没有转义字符",
                            line_number,
                            key,
                        )
                    )
                    segment.append("\\")
                    position += 1
                else:
                    following = text[position + 1]
                    decoded = {
                        "n": "\n",
                        "t": "\t",
                        "r": "\r",
                        "\\": "\\",
                        "'": "'",
                        '"': '"',
                    }.get(following)
                    if decoded is None:
                        self.reasons.append(
                            Reason(
                                "invalid_escape",
                                f"未知转义序列：\\{following}",
                                line_number,
                                key,
                            )
                        )
                        segment.append(text[position : position + 2])
                    else:
                        segment.append(decoded)
                    position += 2
            elif char == quote:
                trailing = text[position + 1 :].strip()
                if trailing:
                    self.reasons.append(
                        Reason(
                            "characters_after_closing_quote",
                                f"闭合引号后存在多余字符：{trailing}",
                                line_number,
                                key,
                        )
                    )
                self.quote_buffer.append("".join(segment))
                return True
            else:
                segment.append(char)
                position += 1

        self.quote_buffer.append("".join(segment))
        return False

    def _store_open_field(self, line_number: int) -> None:
        key = self.open_key or ""
        value = "".join(self.quote_buffer)
        self._reset_quote()
        self._store_field(key, value, line_number)

    def _store_field(self, key: str, value: str, line_number: int) -> None:
        self.field_count += 1
        if key in self.fields:
            self.reasons.append(
                Reason(
                    "duplicate_field",
                    f"字段 {key} 重复出现",
                    line_number,
                    key,
                )
            )
            return
        self.fields[key] = value
        self.field_lines[key] = line_number

    def _validate_fields(self) -> None:
        present = set(self.fields)
        missing = [name for name in REQUIRED_FIELDS if name not in present]
        unexpected = sorted(name for name in present if name not in ALLOWED_FIELDS)

        if self.field_count != len(REQUIRED_FIELDS):
            detail = f"实际有 {self.field_count} 个字段，要求恰好有 {len(REQUIRED_FIELDS)} 个"
            self.reasons.append(
                Reason("field_count_mismatch", detail, self.start_line)
            )

        schema_problems = []
        if missing:
            schema_problems.append(f"缺少字段：{', '.join(missing)}")
        if unexpected:
            schema_problems.append(f"存在未定义字段：{', '.join(unexpected)}")
        if schema_problems:
            self.reasons.append(
                Reason(
                    "schema_mismatch",
                    "；".join(schema_problems),
                    self.start_line,
                )
            )

    def _reset_quote(self) -> None:
        self.open_key = None
        self.open_line = None
        self.open_quote = None
        self.quote_buffer = []


def parse_text(text: str) -> ParseResult:
    return parse_lines(text.splitlines())


def parse_lines(lines: List[str]) -> ParseResult:
    records: List[Dict[str, str]] = []
    bad_records: List[BadRecord] = []
    stream_errors: List[Reason] = []

    state: Optional[RecordState] = None
    preamble_start: Optional[int] = None
    preamble_end: Optional[int] = None

    def flush_preamble() -> None:
        nonlocal preamble_start, preamble_end
        if preamble_start is not None and preamble_end is not None:
            stream_errors.append(
                Reason(
                    "text_outside_record",
                    "第一条记录开始前存在不属于任何记录的文本",
                    preamble_start,
                )
            )
        preamble_start = None
        preamble_end = None

    def close_record() -> None:
        nonlocal state
        if state is None:
            return
        record, bad_record, _fatal = state.finish()
        if record is not None:
            records.append(record)
        if bad_record is not None:
            bad_records.append(bad_record)
        state = None

    for line_number, physical_line in enumerate(lines, start=1):
        if physical_line.strip() == MARKER:
            flush_preamble()
            close_record()
            state = RecordState(line_number)
            continue

        if state is None:
            if physical_line.strip():
                if preamble_start is None:
                    preamble_start = line_number
                preamble_end = line_number
            continue

        state.feed(physical_line, line_number)

    flush_preamble()
    close_record()

    return ParseResult(records, bad_records, stream_errors)


SAMPLE_INPUT = """##RECORD
id=1
name=Alice
email=alice@example.com

##RECORD
id=2
name="Bob
email=bob@example.com
##RECORD
id=3
name="Carol"
email=carol@example.com
note=hello
##RECORD
id=4
name=Dave

##RECORD
this line is broken
id=5
name=Eve
email=eve@example.com
##RECORD
id=6
name='Frank
email=frank@example.com
##RECORD
id=7
name=Grace
email=grace@example.com
"""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="解析多行记录流，报告并跳过坏记录。"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("file", nargs="?", help="待解析的文本文件；省略时从标准输入读取")
    source.add_argument("--demo", action="store_true", help="运行内置输入样例")
    parser.add_argument(
        "--compact",
        action="store_true",
        help="输出单行 JSON；默认使用缩进格式",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.demo:
        text = SAMPLE_INPUT
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as source_file:
            text = source_file.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    else:
        parser.error("请提供输入文件，或使用 --demo，也可以通过标准输入传入文本")

    result = parse_text(text)
    json.dump(
        result.to_dict(),
        sys.stdout,
        ensure_ascii=False,
        indent=None if args.compact else 2,
    )
    sys.stdout.write("\n")
    return 1 if result.bad_records or result.stream_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
