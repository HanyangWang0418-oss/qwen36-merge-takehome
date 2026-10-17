# Expert configs

| config | target benchmark | role |
|---|---|---|
| e1_code_repair | SWE-bench Pro v2-hard | main expert |
| e2_terminal_agent | Terminal-Bench 2.0 + ASI-Bench | main expert (both are sandbox + shell + artifact tasks) |
| e3_reasoning | HLE | main expert |
| e2a_terminal_only, e2b_terminal_swe | e2's data split by task form: pure terminal vs software-engineering-in-terminal | control: 4-expert split vs 3-expert split (the planned research-agent expert had no data; see data/README.md) |
| e1_code_repair_r64 | SWE-bench Pro v2-hard | control: rank 64 (Tinker maximum) vs rank 32 |

Main matrix = {e1, e2, e3} merged. Control A = {e1, e2a, e2b, e3} merged (configs/merge/ta_l0.7_4experts.yaml). Control B = e1_r128 in place of e1.
