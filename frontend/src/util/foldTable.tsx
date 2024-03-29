import React from "react";
import { FaCheckCircle } from "react-icons/fa";
import { describeFoldState, Fold } from "../services/backend.service";
import { Link } from "react-router-dom";

function getTagBadge(tag: string) {
  const badgeStyle = { background: "#999999" };
  return (
    <Link to={`/tag/${tag}`} key={tag}>
      <span className="uk-badge" style={badgeStyle}>
        {tag}
      </span>
    </Link>
  );
}

export function makeFoldTable(folds: Fold[]) {
  return (
    <div className="uk-overflow-auto">
      <table
        className="uk-table uk-table-hover uk-table-small"
        style={{ tableLayout: "fixed" }}
      >
        <thead>
          <tr>
            <th className="uk-width-medium">Name</th>
            <th className="uk-width-small">Length</th>
            <th className="uk-width-small">State</th>
            <th className="uk-width-small">Owner</th>
            <th className="uk-width-small">Date Created</th>
            <th
              className="uk-width-small"
              uk-tooltip={"Whether a fold is visible to the public."}
            >
              Public
            </th>
            <th className="uk-width-small">Docked Ligands</th>
            <th className="uk-width-small">Tags</th>
          </tr>
        </thead>
        <tbody>
          {[...folds].map((fold) => {
            return (
              <tr key={fold.name}>
                <td
                  className="uk-text-truncate"
                  uk-tooltip={`title: ${fold.name}`}
                >
                  <Link to={"/fold/" + fold.id}>
                    <div style={{ height: "100%", width: "100%" }}>
                      {fold.name}
                    </div>
                  </Link>
                </td>
                <td>{fold.sequence.length}</td>
                <td
                  className="uk-text-truncate"
                  uk-tooltip={`title: ${describeFoldState(fold)}`}
                >
                  {/* {foldIsFinished(fold) ? null : <div uk-spinner="ratio: 0.5"></div>}&nbsp;  */}
                  {describeFoldState(fold)}
                </td>
                <td className="uk-text-truncate">{fold.owner}</td>
                <td className="uk-text-truncate">
                  {new Date(fold.create_date).toLocaleString("en-US", {
                    timeStyle: "short",
                    dateStyle: "short",
                  })}
                </td>
                <td className="uk-text-truncate">
                  {fold.public ? (
                    <FaCheckCircle
                      uk-tooltip={"This fold is visible to the public."}
                    />
                  ) : null}
                </td>
                <td
                  className="uk-text-truncate "
                  uk-tooltip={`title: ${(fold.docks || [])
                    .map((d) => d.ligand_name)
                    .slice(0, 5)
                    .join(", ")}`}
                >
                  {(fold.docks || []).length}
                </td>
                <td
                  className="uk-text-nowrap hiddenscrollbar"
                  style={{ overflow: "scroll" }}
                >
                  {[...fold.tags].map(getTagBadge)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
