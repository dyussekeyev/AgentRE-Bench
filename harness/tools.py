from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .sandbox import DockerRunner, PathValidator, RunResult, SubprocessRunner

log = logging.getLogger(__name__)

# ── Anthropic-native tool schemas (canonical format) ──────────────────

TOOL_SCHEMAS = [
    {
        "name": "file",
        "description": "Identify file type. Returns the output of the `file` command.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the binary file (relative to workspace).",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "strings",
        "description": (
            "Extract printable strings from a binary. "
            "Returns readable ASCII/UTF-8 strings found in the file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the binary file.",
                },
                "min_length": {
                    "type": "integer",
                    "description": "Minimum string length (default 4).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "readelf",
        "description": (
            "Display information about ELF binary sections, headers, symbols, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the ELF binary.",
                },
                "flags": {
                    "type": "string",
                    "enum": ["-h", "-S", "-s", "-l", "-d", "-a"],
                    "description": (
                        "readelf flag: -h (header), -S (sections), "
                        "-s (symbols), -l (program headers), "
                        "-d (dynamic), -a (all)."
                    ),
                },
            },
            "required": ["path", "flags"],
        },
    },
    {
        "name": "objdump",
        "description": (
            "Disassemble or dump information from a binary. "
            "Use -d for disassembly, -t for symbols, -x for all headers, -s for full contents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the binary.",
                },
                "flags": {
                    "type": "string",
                    "enum": ["-d", "-D", "-t", "-x", "-s"],
                    "description": (
                        "objdump flag: -d (disassemble), -D (disassemble all), "
                        "-t (symbol table), -x (all headers), -s (full contents)."
                    ),
                },
                "section": {
                    "type": "string",
                    "description": "Optional section name to target (e.g. .text, .rodata).",
                },
            },
            "required": ["path", "flags"],
        },
    },
    {
        "name": "nm",
        "description": "List symbols from an object file or binary.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the binary.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "hexdump",
        "description": (
            "Display a hex+ASCII dump of a binary file. "
            "Useful for examining raw bytes at specific offsets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the binary.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Byte offset to start from (default 0).",
                },
                "length": {
                    "type": "integer",
                    "description": "Number of bytes to dump (max 4096, default 256).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "xxd",
        "description": (
            "Create a hex dump of a file. "
            "Similar to hexdump but with a different output format."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the binary.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Byte offset to start from (default 0).",
                },
                "length": {
                    "type": "integer",
                    "description": "Number of bytes to dump (max 4096, default 256).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "entropy",
        "description": (
            "Compute Shannon entropy (0.0-8.0) over a sliding window. "
            "High entropy (>7.0) indicates encrypted or compressed data. "
            "Low entropy (<4.0) indicates plaintext or sparse data. "
            "Optionally target a specific ELF section."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the binary.",
                },
                "section": {
                    "type": "string",
                    "description": "Optional ELF section name (e.g. .text, .rodata, .data).",
                },
                "window_size": {
                    "type": "integer",
                    "description": "Sliding window size in bytes (default 256).",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "final_answer",
        "description": (
            "Submit your final reverse engineering analysis. "
            "Call this tool ONCE when you have completed your analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_type": {
                    "type": "string",
                    "description": "File format, e.g. 'ELF64'.",
                },
                "encoded_strings": {
                    "type": "boolean",
                    "description": "Whether the binary contains encoded/encrypted strings.",
                },
                "decoded_c2": {
                    "type": "string",
                    "description": "The decoded command-and-control URL or address (e.g. '192.168.1.100:4444' or 'http://example.com/payload').",
                },
                "techniques": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of techniques observed (e.g. 'socket_connect', 'xor_encoding', 'anti_debug_ptrace').",
                },
                "c2_protocol": {
                    "type": "string",
                    "description": "Protocol used for C2 communication (e.g. 'TCP', 'HTTP', 'DNS', 'ICMP').",
                },
                "encryption_details": {
                    "type": "object",
                    "description": "Optional. Encryption details if applicable (algorithm, key, key_storage).",
                    "properties": {
                        "algorithm": {"type": "string"},
                        "key": {"type": "string"},
                        "key_storage": {"type": "string"},
                    },
                },
                "decoded_strings": {
                    "type": "object",
                    "description": "Optional. Dictionary of decoded encrypted strings.",
                },
                "anti_analysis": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional. List of anti-analysis techniques found.",
                },
            },
            "required": ["file_type", "encoded_strings", "decoded_c2", "techniques", "c2_protocol"],
        },
    },
]


# ── Entropy computation script (runs inside sandbox via python3 -c) ───

ENTROPY_SCRIPT = r'''
import math, struct, sys, os

def entropy(data, window=256):
    results = []
    for i in range(0, len(data), window):
        chunk = data[i:i+window]
        if len(chunk) < 16:
            break
        freq = [0]*256
        for b in chunk:
            freq[b] += 1
        n = len(chunk)
        ent = 0.0
        for f in freq:
            if f > 0:
                p = f / n
                ent -= p * math.log2(p)
        results.append((i, len(chunk), round(ent, 4)))
    return results

path = sys.argv[1]
section = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "" else None
window = int(sys.argv[3]) if len(sys.argv) > 3 else 256

with open(path, "rb") as f:
    data = f.read()

if section:
    # Parse ELF section headers to find the section
    if data[:4] != b"\x7fELF":
        print(f"Error: not an ELF file", file=sys.stderr)
        sys.exit(1)
    is_64 = data[4] == 2
    if is_64:
        e_shoff = struct.unpack_from("<Q", data, 40)[0]
        e_shentsize = struct.unpack_from("<H", data, 58)[0]
        e_shnum = struct.unpack_from("<H", data, 60)[0]
        e_shstrndx = struct.unpack_from("<H", data, 62)[0]
        # Get section name string table
        str_sh_off = e_shoff + e_shstrndx * e_shentsize
        str_sh_offset = struct.unpack_from("<Q", data, str_sh_off + 24)[0]
        str_sh_size = struct.unpack_from("<Q", data, str_sh_off + 32)[0]
        strtab = data[str_sh_offset:str_sh_offset+str_sh_size]
        found = False
        for i in range(e_shnum):
            off = e_shoff + i * e_shentsize
            sh_name_idx = struct.unpack_from("<I", data, off)[0]
            name = strtab[sh_name_idx:].split(b"\x00")[0].decode("ascii", errors="replace")
            if name == section:
                sh_offset = struct.unpack_from("<Q", data, off + 24)[0]
                sh_size = struct.unpack_from("<Q", data, off + 32)[0]
                data = data[sh_offset:sh_offset+sh_size]
                found = True
                break
        if not found:
            print(f"Section {section!r} not found", file=sys.stderr)
            sys.exit(1)
    else:
        print("Only ELF64 supported for section targeting", file=sys.stderr)
        sys.exit(1)

results = entropy(data, window)
total_ent = 0.0
if data:
    freq = [0]*256
    for b in data:
        freq[b] += 1
    n = len(data)
    for f in freq:
        if f > 0:
            p = f / n
            total_ent -= p * math.log2(p)

print(f"Total size: {len(data)} bytes")
print(f"Overall entropy: {total_ent:.4f} bits/byte")
print(f"Window size: {window} bytes")
print(f"Windows analyzed: {len(results)}")
print()
if results:
    ents = [r[2] for r in results]
    print(f"Min window entropy: {min(ents):.4f}")
    print(f"Max window entropy: {max(ents):.4f}")
    print(f"Avg window entropy: {sum(ents)/len(ents):.4f}")
    print()
    print("Offset      Size  Entropy")
    print("-" * 35)
    for offset, size, ent in results[:50]:
        bar = "#" * int(ent * 4)
        print(f"0x{offset:08x}  {size:4d}  {ent:.4f}  {bar}")
    if len(results) > 50:
        print(f"... ({len(results) - 50} more windows)")
'''


# ── Tool execution ────────────────────────────────────────────────────

class ToolExecutor:
    def __init__(self, config: BenchmarkConfig, binary_path: Path):
        self.config = config
        self.binary_path = binary_path.resolve()
        self.validator = PathValidator(config.workspace_dir)

        if config.use_docker:
            self.runner = DockerRunner(
                image=config.docker_image,
                workspace_dir=config.workspace_dir,
                timeout=config.tool_timeout_seconds,
                max_output_chars=config.max_output_chars,
            )
        else:
            self.runner = SubprocessRunner(
                workspace_dir=config.workspace_dir,
                timeout=config.tool_timeout_seconds,
                max_output_chars=config.max_output_chars,
            )

    def _resolve_path(self, path_arg: str) -> str:
        # The agent may send paths like "/workspace/binary" (Docker-style)
        # or just "binary". Strip the /workspace/ prefix before validating
        # against the real workspace directory.
        clean = path_arg
        if clean.startswith("/workspace/"):
            clean = clean[len("/workspace/"):]
        elif clean.startswith("/workspace"):
            clean = clean[len("/workspace"):]

        validated = self.validator.validate(clean)
        if self.config.use_docker:
            return "/workspace/" + str(validated.relative_to(self.config.workspace_dir))
        return str(validated)

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "final_answer":
            return {"is_final_answer": True, "answer": tool_input}

        if tool_name not in self.config.allowed_tools:
            return {
                "is_final_answer": False,
                "error": f"Tool {tool_name!r} is not allowed.",
            }

        try:
            cmd = self._build_command(tool_name, tool_input)
        except (ValueError, FileNotFoundError) as e:
            return {"is_final_answer": False, "error": str(e)}

        result = self.runner.run(cmd)
        return self._format_result(result)

    def _build_command(self, tool_name: str, args: dict[str, Any]) -> list[str]:
        path = self._resolve_path(args.get("path", ""))

        if tool_name == "file":
            return ["file", path]

        if tool_name == "strings":
            cmd = ["strings"]
            ml = args.get("min_length")
            if ml is not None:
                cmd += ["-n", str(int(ml))]
            cmd.append(path)
            return cmd

        if tool_name == "readelf":
            flags = args.get("flags", "-h")
            if flags not in ("-h", "-S", "-s", "-l", "-d", "-a"):
                raise ValueError(f"Invalid readelf flag: {flags!r}")
            return ["readelf", flags, path]

        if tool_name == "objdump":
            flags = args.get("flags", "-d")
            if flags not in ("-d", "-D", "-t", "-x", "-s"):
                raise ValueError(f"Invalid objdump flag: {flags!r}")
            cmd = ["objdump", flags]
            section = args.get("section")
            if section:
                cmd += ["-j", str(section)]
            cmd.append(path)
            return cmd

        if tool_name == "nm":
            return ["nm", path]

        if tool_name == "hexdump":
            offset = args.get("offset", 0)
            length = min(args.get("length", 256), 4096)
            return ["hexdump", "-C", "-s", str(int(offset)), "-n", str(int(length)), path]

        if tool_name == "xxd":
            offset = args.get("offset", 0)
            length = min(args.get("length", 256), 4096)
            return ["xxd", "-s", str(int(offset)), "-l", str(int(length)), path]

        if tool_name == "entropy":
            section = args.get("section", "")
            window = str(args.get("window_size", 256))
            return [
                "python3", "-c", ENTROPY_SCRIPT,
                path, section, window,
            ]

        raise ValueError(f"Unknown tool: {tool_name!r}")

    def _format_result(self, result: RunResult) -> dict[str, Any]:
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr] {result.stderr}")
        if result.timed_out:
            output_parts.append("[timed out]")
        if result.truncated:
            output_parts.append("[output was truncated]")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        return {
            "is_final_answer": False,
            "output": output,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "truncated": result.truncated,
        }


def get_tool_schemas(include_final_answer: bool = True) -> list[dict]:
    if include_final_answer:
        return list(TOOL_SCHEMAS)
    return [t for t in TOOL_SCHEMAS if t["name"] != "final_answer"]


def schemas_to_openai(schemas: list[dict]) -> list[dict]:
    tools = []
    for s in schemas:
        tools.append({
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        })
    return tools


def schemas_to_gemini_declarations(schemas: list[dict]) -> list[dict]:
    declarations = []
    for s in schemas:
        declarations.append({
            "name": s["name"],
            "description": s["description"],
            "parameters": s["input_schema"],
        })
    return declarations
