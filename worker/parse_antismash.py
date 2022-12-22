#!/usr/bin/env python
""" This CLI tool is meant to be used to parse antismash json files into
list of domain annotations"""
__VERSION__ = "0.1"
__AUTHOR__ = "Alberto Nava <alberto_nava@berkeley.edu>"
__LASTUPDATED__ = "220407"

# =======================================================================#
# Importations
# =======================================================================#

# Built-in Python Libraries
import os
import json
import logging
import argparse
from typing import List, Dict, Any
from pathlib import Path


# =======================================================================#
# Logic
# =======================================================================#


class NotAPKSRecord(Exception):
    pass


def parse_antismash_record(antismash_record: dict) -> List[dict]:
    logging.debug(f'Analyzing record {antismash_record["id"]}')

    if (
        ("modules" not in antismash_record.keys())
        or (
            "antismash.detection.nrps_pks_domains"
            not in antismash_record["modules"].keys()
        )
        or (
            "cds_results"
            not in antismash_record["modules"][
                "antismash.detection.nrps_pks_domains"
            ].keys()
        )
        or (
            len(
                list(
                    antismash_record["modules"][
                        "antismash.detection.nrps_pks_domains"
                    ]["cds_results"].keys()
                )
            )
            != 1
        )
    ):
        raise NotAPKSRecord
    gene_id: str = list(
        antismash_record["modules"][
            "antismash.detection.nrps_pks_domains"
        ]["cds_results"].keys()
    )[0]
    return [
        {
            "type": domain["hit_id"],
            "start": domain["query_start"],
            "end": domain["query_end"],
        }
        for domain in antismash_record["modules"][
            "antismash.detection.nrps_pks_domains"
        ]["cds_results"][gene_id]["domain_hmms"]
    ]


def parse_antismash_json(antismash_json: dict) -> dict:
    """
    Parse json <- record <- region <- gene <- module <- domain
    """
    logging.debug(
        f"Begin parsing {len(antismash_json['records'])} records"
        f" for {antismash_json['records'][0]['description']}"
    )

    pks_bgcs: dict = {}
    for record in antismash_json["records"]:
        try:
            pks_bgcs[record["id"]] = parse_antismash_record(
                antismash_record=record
            )
        except NotAPKSRecord:
            logging.debug(f"Record: {record['id']} not a PKS record")

    return pks_bgcs


def run_single_parser(
    antismash_json_file: Path,
    simple_output_filename: Path,
) -> dict:
    simple_output_filename.parent.mkdir(exist_ok=True)

    logging.debug(f"Begin parsing {antismash_json_file} ...")
    with open(antismash_json_file, "r") as F:
        antismash_json = json.load(F)

    parsed_antismash = parse_antismash_json(antismash_json=antismash_json)
    with open(simple_output_filename, "w") as F:
        json.dump(parsed_antismash, F, indent=4)
    print(parsed_antismash)

    logging.debug(f"Finished parsing {antismash_json_file}")
    return parsed_antismash


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
        description=("A CLI for parsing antismash json files"),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "json_dir",
        type=str,
        help=("Path to directory containing antismash json file"),
    )
    parser.add_argument(
        "simple_output_filename",
        type=str,
        help=(
            "Path to file where simplified parsed json file will be"
            " stored"
        ),
    )
    return parser


def loggingHelper(
    verbose: bool = False, filename: str = "antismash_parser_log.log"
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
    assert os.path.isdir(args["json_dir"])

    logging.debug("Starting to parse")

    json_files = list(Path(args["json_dir"]).glob("*.json"))
    assert (
        len(json_files) == 1
    ), f"Need 1 json file in antismash directory, got {len(json_files)}"
    run_single_parser(
        antismash_json_file=json_files[0],
        simple_output_filename=Path(args["simple_output_filename"]),
    )

    logging.debug("Finished parsing")
    return None


if __name__ == "__main__":
    args = vars(cli().parse_args())
    loggingHelper(verbose=args["verbose"])
    main(args)
