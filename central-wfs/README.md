### Intro
Project architecture to centrally ingest CHoRUS site waveforms; identify and correct gaps; and convert to WFDB format.

Layout mimics extensive work done [here](https://github.com/chorus-ai/chorus_waveform/tree/main) by some wonderful people.

Below diagrams how waveform data flows with this framework, starting at site upload.

```mermaid
---
title: Ingestion-Conversion pipeline
---

flowchart TD

    A@{ shape: stadium, label: "__main__: CLI call options, summary record(s) table" }

    Z@{ shape: stadium, label: "assemble_waveforms: inventory waveform file paths" }

    B@{ shape: stadium, label: "run_benchmarks(waveform_suite_table)" }

    Y@{ shape: stadium, label: "load_signals: site/format-dependent, may be WFDB already" }

    C@{ shape: stadium, label: "waveform_summary: per channel, per chunk" }

    W@{shape: delay, label: "process_gaps: identify and convert" }

    D@{ shape: stadium, label: "write_waveforms: site/format-dependent, to WFDB" }

    E@{ shape: stadium, label: "fidelity checks: data quality, read/write operations" }

    X@{ shape: rounded, label: "site uploads data" }

    F@{ shape: database, label: "save_output_to_log: store converted metadata and data" }

    G@{ shape: rounded, label: "PerformanceCounter: memory profiler" }

    H@{ shape: lin-rect, label: "formats/ and sites/: import specific functions and methods" }

    I@{ shape: lin-rect, label: "registry: define and locate site configurations"}

    %%LINKS
    A --> Z --> B --> operations

    subgraph operations
        direction TB
        subgraph read/write
            direction TB
            Y --> C --> D
        end
        G --> read/write
        read/write --> E
        I --> H --> read/write
    end
    X XZ@--> Z
    operations oF@--> F
    W -.-> read/write

    %%STYLING
    oF@{ animation: fast}
    XZ@{ animation: slow}
```

### Repo layout
```
central_wfs/
├── config/ # setup
│ ├── .env
│ └── environment.yml # conda/mamba install
├── src/ # Source code
│   ├── qc/
│   │   ├── chs_norm.py # check channel mappings
│   │   ├── data.py # fidelity check on data quality
│   │   ├── gaps.py # identify and correct gaps
│   │   └── ioperf.py # monitor memory & performance
│   ├── formats/
│   │   ├── map.py # dictionary to standardize site file extensions
│   │   ├── csv.py
│   │   ├── parquet.py
│   │   ├── wfdb.py
│   │   ├── format_n.py
│   ├── sites/
│   │   ├── uf.py
│   │   ├── columbia.py
│   │   ├── mgh.py
│   │   ├── mit.py
│   │   └── site_n.py
│   ├── __main__.py # entry point
│   ├── __init__.py
│   ├── base.py # define site/format-specific class requirements
│   ├── registry.py # define site configs
│   └── utils.py
├── data/
│   └── suites/ # file path tables
├── tests/ # unit/integration tests
├── docs/ # documentation
├── .gitignore
├── pyproject.toml # pip install
├── requirements.txt # python dependencies
└── README.md # project overview
```

#### Design notes
1. Making gap tooling central versus site-specific (currently central).
2. Abstracting site AND format class methods in preparation for potential heterogeneity of site waveform uploads.
3. Additional modifications were denoted in the neighborhood with '###'
4. Site-specific classes inherit from format-specific classes, and include a method for assembling the waveform suite table.