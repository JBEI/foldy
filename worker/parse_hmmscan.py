#!/usr/bin/env python

"""This module is used to parse domtblout from an hmmscan performed to
annotate pfam domains in a protein sequence"""
__VERSION__ = "0.1"
__AUTHOR__ = "Alberto Nava <alberto_nava@berkeley.edu>"
__LASTUPDATED__ = "220420"

# =======================================================================#
# Importations
# =======================================================================#

# Built-in Python Libraries
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path
from Bio import SearchIO

# =======================================================================#
# Logic
# =======================================================================#


def domain_overlaps(
    query_domain: Dict[str, Any], existing_domains: List[Dict[str, Any]]
) -> bool:
    for existing_domain in existing_domains:
        if (
            (query_domain["start"] > existing_domain["start"])
            and (query_domain["start"] < existing_domain["end"])
        ) or (
            (query_domain["end"] > existing_domain["start"])
            and (query_domain["end"] < existing_domain["end"])
        ):
            return True
    return False


def parse_hmmscan(input_file: Path) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}
    hmmscan_result = SearchIO.parse(input_file, "hmmscan3-domtab")
    for query_result in hmmscan_result:
        results[query_result.id] = []
        for hit in query_result:
            for hsp in hit:
                domain: Dict[str, Any] = {
                    "start": hsp.query_start,
                    "end": hsp.query_end,
                    "type": hit.id,
                }
                if not domain_overlaps(
                    query_domain=domain,
                    existing_domains=results[hit.query_id],
                ):
                    results[hit.query_id].append(domain)
    return {
        chain: sorted(results[chain], key=lambda obj: obj["start"])
        for chain in results
    }


# =======================================================================#
# Command-Line Interface
# =======================================================================#


def cli() -> argparse.ArgumentParser:
    """Command-Line Interface Function

    Arguments
    ---------
    None
    Returns
    -------
    parser : argparse.ArgumentParser
        A command line parser object
    """
    parser = argparse.ArgumentParser(
        description=("A CLI for parsing hmmscan domtblout"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "input",
        type=str,
        help=("Path to hmmscan domtblout file to parse"),
    )
    parser.add_argument(
        "output",
        type=str,
        help=("Path to file where output json file will be stored"),
    )
    return parser


def loggingHelper(
    verbose: bool = False, filename: str = "hmmscan_parser_log.log"
) -> None:
    """Helper to set up python logging

    Arguments
    ---------
    verbose : bool, optional
        Whether to set up verbose logging [default: False]

    Returns
    -------
    None

        Sets up logging
    """
    if verbose:
        loggingLevel = logging.DEBUG  # show everything
    else:
        loggingLevel = logging.ERROR  # show only ERROR and CRITICAL
    logging.basicConfig(
        filename=filename,
        level=loggingLevel,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    return None


# =======================================================================#
# Main
# =======================================================================#


def main(args: Dict[str, Any]) -> None:
    """Command-Line Main Function

    Arguments
    ---------
    args : dict
        CLI-interface options from argparse

    Returns
    -------
    None
    """
    logging.debug(args)

    input_file: Path = Path(args["input"])
    output_file: Path = Path(args["output"])

    assert input_file.exists()

    logging.debug("Starting to parse")

    annotations: Dict[str, List[Dict[str, Any]]] = parse_hmmscan(
        input_file=input_file
    )
    with open(output_file, "w") as F:
        json.dump(annotations, F, indent=4)

    logging.debug("Finished parsing")
    return None


if __name__ == "__main__":
    args = vars(cli().parse_args())
    loggingHelper(verbose=args["verbose"])
    main(args)
