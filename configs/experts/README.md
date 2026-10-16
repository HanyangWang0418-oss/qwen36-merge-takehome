# Expert configs

| config | target benchmark | role |
|---|---|---|
| e1_code_repair | SWE-bench Pro v2-hard | main expert |
| e2_terminal_agent | Terminal-Bench 2.0 + ASI-Bench | main expert (both are sandbox + shell + artifact tasks) |
| e3_reasoning | HLE | main expert |
| e2a_terminal_only, e2b_research_agent | TB2 / ASI separately | control: 4-expert split vs 3-expert split |
| e1_code_repair_r128 | SWE-bench Pro v2-hard | control: rank 128 vs rank 32 |

Main matrix = {e1, e2, e3} merged. Control A = {e1, e2a, e2b, e3} merged. Control B = e1_r128 in place of e1.
