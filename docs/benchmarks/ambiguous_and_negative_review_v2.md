# Ambiguous And Negative Review V2

Status date: 2026-07-01.

## Ambiguous Queries

No query is labelled `ambiguous` in `benchmark_queries_v2.json`. Multi-answer positive queries keep multiple expected views/nodes instead of forcing one canonical answer.

Freshness/change queries are not included because the captured scenes have no temporal recapture or change labels. They require new temporal evidence rather than manual invention.

## Negative Query Scope

Negative cases are verified absent only from the manual ViewJSON semantic index for the target captured scene. They are not independent visual absence proofs.

| Query ID | Scene | Query | Negative label | Verification source |
|---|---|---|---|---|
| `qv2_026` | `ConferenceHall` | Find bed | bed | `backend/data/scenes/ConferenceHall-capture-pilot/views/` |
| `qv2_027` | `ConferenceHall` | Find shower | shower | `backend/data/scenes/ConferenceHall-capture-pilot/views/` |
| `qv2_028` | `ConferenceHall` | Find vehicle | vehicle | `backend/data/scenes/ConferenceHall-capture-pilot/views/` |
| `qv2_029` | `ConferenceHall` | Find swimming pool | swimming pool | `backend/data/scenes/ConferenceHall-capture-pilot/views/` |
| `qv2_030` | `ConferenceHall` | Find kitchen stove | kitchen stove | `backend/data/scenes/ConferenceHall-capture-pilot/views/` |
| `qv2_056` | `Museume` | Find grand piano | grand piano | `backend/data/scenes/Museume-capture/views/` |
| `qv2_057` | `Museume` | Find swimming pool | swimming pool | `backend/data/scenes/Museume-capture/views/` |
| `qv2_058` | `Museume` | Find bed | bed | `backend/data/scenes/Museume-capture/views/` |
| `qv2_059` | `Museume` | Find shower | shower | `backend/data/scenes/Museume-capture/views/` |
| `qv2_060` | `Museume` | Find projection screen | projection screen | `backend/data/scenes/Museume-capture/views/` |
| `qv2_086` | `Theater` | Find swimming pool | swimming pool | `backend/data/scenes/Theater-capture/views/` |
| `qv2_087` | `Theater` | Find bicycle | bicycle | `backend/data/scenes/Theater-capture/views/` |
| `qv2_088` | `Theater` | Find kitchen stove | kitchen stove | `backend/data/scenes/Theater-capture/views/` |
| `qv2_089` | `Theater` | Find fortress island | fortress island | `backend/data/scenes/Theater-capture/views/` |
| `qv2_090` | `Theater` | Find beach | beach | `backend/data/scenes/Theater-capture/views/` |
| `qv2_116` | `outdoor-street` | Find grand piano | grand piano | `backend/data/scenes/outdoor-street-capture/views/` |
| `qv2_117` | `outdoor-street` | Find swimming pool | swimming pool | `backend/data/scenes/outdoor-street-capture/views/` |
| `qv2_118` | `outdoor-street` | Find bed | bed | `backend/data/scenes/outdoor-street-capture/views/` |
| `qv2_119` | `outdoor-street` | Find sofa | sofa | `backend/data/scenes/outdoor-street-capture/views/` |
| `qv2_120` | `outdoor-street` | Find open sea water | open sea water | `backend/data/scenes/outdoor-street-capture/views/` |
| `qv2_146` | `outdoor-drone` | Find grand piano | grand piano | `backend/data/scenes/outdoor-drone-capture/views/` |
| `qv2_147` | `outdoor-drone` | Find bed | bed | `backend/data/scenes/outdoor-drone-capture/views/` |
| `qv2_148` | `outdoor-drone` | Find shower | shower | `backend/data/scenes/outdoor-drone-capture/views/` |
| `qv2_149` | `outdoor-drone` | Find traffic signs | traffic signs | `backend/data/scenes/outdoor-drone-capture/views/` |
| `qv2_150` | `outdoor-drone` | Find theater stage | theater stage | `backend/data/scenes/outdoor-drone-capture/views/` |
