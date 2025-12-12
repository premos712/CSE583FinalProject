#!/usr/bin/env python3
import csv
import sys

def die(msg: str) -> None:
    print(f"[join_bl.py] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)

def read_dynamic_profile(path: str):
    """
    bl_profile.txt:
      function,path_id,count
      kernel_2mm,0,1655999999
      ...
    Returns: dict[(function, path_id)] = count (int)
    """
    prof = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            die(f"{path} has no header")
        need = {"function", "path_id", "count"}
        if not need.issubset(set(r.fieldnames)):
            die(f"{path} missing columns, got {r.fieldnames}, need {sorted(need)}")

        for row in r:
            fn = row["function"].strip()
            pid = row["path_id"].strip()
            cnt = row["count"].strip()
            if fn == "" or pid == "" or cnt == "":
                continue
            try:
                pid_i = int(pid)
                cnt_i = int(cnt)
            except ValueError:
                continue
            prof[(fn, pid_i)] = cnt_i
    return prof

def join(static_csv: str, bl_profile: str, out_csv: str):
    prof = read_dynamic_profile(bl_profile)

    with open(static_csv, "r", newline="") as fin:
        r = csv.DictReader(fin)
        if r.fieldnames is None:
            die(f"{static_csv} has no header")

        need = {"program", "function", "path_id"}
        if not need.issubset(set(r.fieldnames)):
            die(f"{static_csv} missing columns, got {r.fieldnames}, need {sorted(need)}")

        static_fields = list(r.fieldnames)

        # ---- define your ORIGINAL desired order (no dyn_count) ----
        prefix = [
            "program",
            "function",
            "path_index",   # your original DumpProfilePass uses path_index
            "is_hot",
            "path_len",
            "path_ir",
            "inst_count",
            "branch_count",
            "call_count",
            "loop_depth",
            "in_loop",
            "num_succ",
            "num_preds",
            "dist_from_entry",
            "dom_depth",
            "int_operands",
            "fp_operands",
            "ptr_operands",
            "vector_operands",
            "phi_incoming",
        ]

        # static.csv uses path_id (Ball-Larus), but your original expects path_index.
        # We'll map path_id -> path_index in output.
        # If static.csv already has path_index, we keep it.
        has_path_index = "path_index" in static_fields
        has_path_id = "path_id" in static_fields

        # opcode columns at end
        opcode_cols = [c for c in static_fields if c.startswith("op_")]

        # build final out fields:
        out_fields = []
        out_fields.append("program")
        out_fields.append("function")
        out_fields.append("path_index")
        out_fields.append("is_hot")

        # the rest of prefix (skip ones already added)
        for c in prefix:
            if c in ("program", "function", "path_index", "is_hot"):
                continue
            if c in static_fields:
                out_fields.append(c)

        # opcode columns last
        out_fields += opcode_cols

        # sanity: required columns must exist in static or be derivable
        for req in ["program", "function"]:
            if req not in static_fields:
                die(f"{static_csv} missing required column {req}")

        if not has_path_index and not has_path_id:
            die(f"{static_csv} must contain path_index or path_id")

        # write output
        with open(out_csv, "w", newline="") as fout:
            w = csv.DictWriter(fout, fieldnames=out_fields)
            w.writeheader()

            for row in r:
                fn = row["function"].strip()

                # path index source: prefer path_index, else use path_id
                pid_val = None
                if has_path_index:
                    pid_val = row.get("path_index", "").strip()
                else:
                    pid_val = row.get("path_id", "").strip()

                try:
                    pid = int(pid_val)
                except ValueError:
                    continue

                cnt = prof.get((fn, pid), 0)
                is_hot = "1" if cnt > 0 else "0"

                out_row = {}
                out_row["program"] = row.get("program", "")
                out_row["function"] = row.get("function", "")
                out_row["path_index"] = str(pid)
                out_row["is_hot"] = is_hot

                # copy remaining columns in out_fields
                for c in out_fields:
                    if c in out_row:
                        continue
                    if c in row:
                        out_row[c] = row[c]
                    else:
                        out_row[c] = ""  # should not happen for known cols

                w.writerow(out_row)

def main():
    if len(sys.argv) != 4:
        print("Usage: join_bl.py <static.csv> <bl_profile.txt> <out.csv>", file=sys.stderr)
        sys.exit(2)

    static_csv, bl_profile, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
    join(static_csv, bl_profile, out_csv)

if __name__ == "__main__":
    main()
