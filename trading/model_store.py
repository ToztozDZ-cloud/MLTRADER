from __future__ import annotations

import os
import pickle
import tempfile

from agents.agent import Agent


def save_agent(
    agent: Agent,
    path: str,
) -> None:

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    # Write atomically so the paper trader never sees
    # a partially-written pickle file.

    fd, temp_path = tempfile.mkstemp(
        dir=directory or ".",
        suffix=".tmp",
    )

    try:
        with os.fdopen(
            fd,
            "wb",
        ) as file:

            pickle.dump(
                agent,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        os.replace(
            temp_path,
            path,
        )

    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass

        raise


def load_agent(
    path: str,
) -> Agent:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Agent file not found: {path}"
        )

    with open(
        path,
        "rb",
    ) as file:

        agent = pickle.load(file)

    if not isinstance(
        agent,
        Agent,
    ):
        raise TypeError(
            "Saved object is not an Agent."
        )

    # A paper-trading snapshot must start with
    # a clean runtime state.

    agent.reset()

    return agent