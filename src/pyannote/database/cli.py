#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020- CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Alexis PLAQUET

import math
import os
from enum import Enum
from pathlib import Path
from typing import Optional, Text

import boto3
import typer
import yaml
from pyannote.core import Annotation
from typing_extensions import Annotated

from pyannote.database import Database, FileFinder, registry
from pyannote.database.protocol import CollectionProtocol, SpeakerDiarizationProtocol

app = typer.Typer()


class Task(str, Enum):
    Any = "Any"
    Protocol = "Protocol"
    Collection = "Collection"
    SpeakerDiarization = "SpeakerDiarization"
    SpeakerVerification = "SpeakerVerification"


@app.command("database")
def database():
    """Print list of databases"""
    for database in registry.databases:
        typer.echo(f"{database}")


@app.command("task")
def task(
    database: str = typer.Option(
        "",
        "--database",
        "-d",
        metavar="DATABASE",
        help="Filter tasks by DATABASE.",
        case_sensitive=False,
    )
):
    """Print list of tasks"""

    if database == "":
        tasks = []
    else:
        db: Database = registry.get_database(database)
        tasks = db.get_tasks()

    for task in tasks:
        typer.echo(f"{task}")


@app.command("protocol")
def protocol(
    database: str = typer.Option(
        "",
        "--database",
        "-d",
        metavar="DATABASE",
        help="Filter protocols by DATABASE.",
        case_sensitive=False,
    ),
    task: Task = typer.Option(
        "Any",
        "--task",
        "-t",
        help="Filter protocols by TASK.",
        case_sensitive=False,
    ),
):
    """Print list of protocols"""

    if database == "":
        databases = list(registry.databases)
    else:
        databases = [database]

    for database_name in databases:
        db: Database = registry.get_database(database_name)
        tasks = db.get_tasks() if task == "Any" else [task]
        for task_name in tasks:
            try:
                protocols = db.get_protocols(task_name)
            except KeyError:
                continue
            for protocol in protocols:
                typer.echo(f"{database_name}.{task_name}.{protocol}")


def duration_to_str(seconds: float) -> Text:
    hours = math.floor(seconds / 3600)
    minutes = math.floor((seconds - 3600 * hours) / 60)
    return f"{hours}h{minutes:02d}m"


@app.command("info")
def info(protocol: str):
    """Print protocol detailed information"""

    p = registry.get_protocol(protocol)

    if isinstance(p, SpeakerDiarizationProtocol):
        subsets = ["train", "development", "test"]
        skip_annotation = False
        skip_annotated = False
    elif isinstance(p, CollectionProtocol):
        subsets = ["files"]
        skip_annotation = True
        skip_annotated = True
    else:
        typer.echo("Only collections and speaker diarization protocols are supported.")
        typer.Exit(code=1)

    for subset in subsets:

        num_files = 0
        speakers = set()
        duration = 0.0
        speech = 0.0

        def iterate():
            try:
                for file in getattr(p, subset)():
                    yield file
            except (AttributeError, NotImplementedError):
                return

        for file in iterate():
            num_files += 1

            if not skip_annotation:
                annotation = file["annotation"] or Annotation(uri=file["uri"])
                speakers.update(annotation.labels())
                speech += annotation.get_timeline().support().duration()

            if not skip_annotated:
                annotated = file["annotated"]
                duration += annotated.duration()

        if num_files > 0:
            typer.secho(
                f"{subset}", fg=typer.colors.BRIGHT_GREEN, underline=True, bold=True
            )
            typer.echo(f"   {num_files} files")
            if not skip_annotated:
                typer.echo(f"   {duration_to_str(duration)} annotated")

            if not skip_annotation:
                typer.echo(
                    f"   {duration_to_str(speech)} of speech ({100 * speech / duration:.0f}%)"
                )
                typer.echo(f"   {len(speakers)} speakers")


@app.command("upload", help="Upload protocol(s) to AWS S3")
def upload(
    dataset: Annotated[
        Path,
        typer.Argument(help="Path to dataset to upload"),
    ],
    s3_path: Annotated[
        str,
        typer.Argument(help="S3 path to upload the protocol on, at bucket/path/to/dataset format."),
    ],
    targets: Annotated[
        list[str],
        typer.Option(
            "--protocol",
            "-p",
            case_sensitive=False,
            metavar="TARGETS",
            help="Target protocols to upload on S3, at database.task.protocol.subset format",
        ),
    ] = ["*"],
    database: Annotated[
        str,
        typer.Option(
            "--database",
            "-d",
            metavar="DATABASE",
            help="Path to database.yml. By default, the command will try to find a database.yml file into dataset_root",
        ),
    ] = "",
):
    """Upload dataset to AWS S3."""
    dataset = dataset.resolve()

    s3_database = {
        "Databases": {},
        "Protocols": {},
    }

    s3 = boto3.client("s3")

    splitted_s3_path = s3_path.split("/")
    bucket = splitted_s3_path[0]
    try:
        s3_dataset = "/".join(splitted_s3_path[1:])
    except IndexError:
        s3_dataset = dataset.name

    # if not database was specified, try to find one in dataset repo
    if not database:
        typer.echo(f"No database.yml file specified. Looking for one in {dataset}")
        for dirpath, _, filenames in os.walk(dataset):
            for filename in filenames:
                if filename == "database.yml":
                    database = f"{dirpath}/{filename}"
                    typer.echo(f"Database file found at {database}.")
                    break

    if not database:
        raise FileNotFoundError("No database.yml found")
    database_path = Path(database).resolve()

    registry.load_database(database_path)
    with open(database_path, "r") as stream:
        input_db = yaml.safe_load(stream)

    for target in targets:
        typer.echo(f"Processing {target}...")
        target = target.split(".")

        sdatabase = target[0]
        stask = target[1] if len(target) > 1 else "*"
        sprotocol = target[2] if len(target) > 2 else "*"
        ssubset = target[3] if len(target) > 3 else "*"

        databases = (
            list(registry.databases) if sdatabase == "*" else sdatabase.split("|")
        )
        for database_name in databases:
            db = registry.get_database(database_name)
            s3_database["Databases"][database_name] = input_db["Databases"][
                database_name
            ]
            s3_database["Protocols"][database_name] = {}

            tasks = db.get_tasks() if stask == "*" else stask.split("|")
            for task_name in tasks:
                try:
                    protocols = db.get_protocols(task_name)
                except KeyError:
                    continue

                s3_database["Protocols"][database_name][task_name] = {}

                if sprotocol != "*":
                    protocols = [p for p in protocols if p in sprotocol.split("|")]

                for protocol_name in protocols:
                    typer.echo(
                        f"Processing {database_name}.{task_name}.{protocol_name}..."
                    )

                    # load current protocol
                    protocol = registry.get_protocol(
                        f"{database_name}.{task_name}.{protocol_name}",
                        preprocessors={"audio": FileFinder()},
                    )

                    input_db_protocol: dict = input_db["Protocols"][database_name][
                        task_name
                    ][protocol_name]

                    protocol_metadata = {}
                    protocol_metadata["scope"] = input_db_protocol.pop("scope", "file")

                    subsets = (
                        list(input_db_protocol.keys())
                        if ssubset == "*"
                        else ssubset.split("|")
                    )

                    for subset in subsets:
                        typer.echo(f"Processing {subset}...")

                        protocol_metadata[subset] = {}

                        paths_to_upload: list[str] = []
                        for key, path in input_db_protocol[subset].items():
                            protocol_metadata[subset][key] = path

                            # if path contains uri placeholder
                            if "{uri}" in path:
                                paths_to_upload.append(path)
                                continue

                            path = Path(path)

                            if path.is_absolute():
                                abspath = path
                            else:
                                abspath = database_path.parent / path

                            relpath = os.path.relpath(abspath, dataset)
                            s3.upload_file(
                                str(abspath), bucket, f"{s3_dataset}/{relpath}"
                            )

                        for protocol_file in getattr(protocol, subset)():
                            audio = protocol_file["audio"]
                            relaudio = os.path.relpath(audio, dataset)
                            uri = protocol_file["uri"]

                            s3.upload_file(audio, bucket, f"{s3_dataset}/{relaudio}")
                            for path in paths_to_upload:
                                path = Path(path.format(uri=uri))

                                if path.is_absolute():
                                    abspath = path
                                else:
                                    abspath = database_path.parent / path

                                relpath = os.path.relpath(abspath, dataset)
                                s3.upload_file(
                                    str(abspath), bucket, f"{s3_dataset}/{relpath}"
                                )

                    s3_database["Protocols"][database_name][task_name][
                        protocol_name
                    ] = protocol_metadata

    s3_database_path = Path("s3_database.yml")
    with open(s3_database_path, "w") as stream:
        yaml.safe_dump(s3_database, stream)

    s3.upload_file(
        str(s3_database_path),
        bucket,
        f"{s3_dataset}/{os.path.relpath(database_path, dataset)}",
    )
    s3_database_path.unlink()


def main():
    app()
