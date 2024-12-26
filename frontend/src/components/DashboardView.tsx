import React, { FormEvent, useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  authenticationService,
  DecodedJwt,
} from "../services/authentication.service";
import { Fold, getFolds } from "../services/backend.service";
import { makeFoldTable } from "../util/foldTable";
import qs from "query-string";
// var debounce = require("lodash/debounce");
import debounce from "lodash/debounce";

const PAGE_SIZE = 50;

const setQueryStringWithoutPageReload = (qsValue: string) => {
  const newurl =
    window.location.protocol +
    "//" +
    window.location.host +
    window.location.pathname +
    qsValue;

  window.history.pushState({ path: newurl }, "", newurl);
};

const setQueryStringValue = (
  key: string,
  value: string,
  queryString = window.location.search
) => {
  const values = qs.parse(queryString);
  const newQsValue = qs.stringify({ ...values, [key]: value });
  setQueryStringWithoutPageReload(`?${newQsValue}`);
};

function getQueryStringValue(
  key: string,
  queryString = window.location.search
): string[] {
  const values = qs.parse(queryString);
  const value: null | string | (string | null)[] = values[key];
  if (typeof value === "string") {
    return [value];
  }
  if (!value) {
    return [];
  }
  return value.map((e) => e || "");
}

function AuthenticatedDashboardView(props: {
  setErrorText: (a: string) => void;
  decodedToken: DecodedJwt;
}) {
  const userEmail: string = props.decodedToken.user_claims.email;

  const filterQueryString = getQueryStringValue("filter");
  const [page, rawSetPage] = useState<string[]>(getQueryStringValue("page"));
  const [filter, setFilter] = useState<string[]>(
    filterQueryString.length !== 0 && filterQueryString[0]
      ? filterQueryString
      : [userEmail]
  );
  const [filterFormValue, setFilterFormValue] = useState<string>(filter[0]);
  const [folds, setFolds] = useState<Fold[] | null>(null);
  const [searchIsStale, setSearchIsStale] = useState<boolean>(false);

  const setPage = useCallback((newValue: number) => {
    rawSetPage([`${newValue}`]);
    setQueryStringValue("page", `${newValue}`);
  }, []);
  // const setFilter = useCallback(
  //   (newValue) => {
  //     setPage(1);
  //     rawSetFilter(newValue);
  //     setQueryStringValue('filter', newValue);
  //   },
  //   [setPage]
  // );

  const pageNum = page.length !== 0 ? parseInt(page[0]) : 1;

  // Note, a few random stack overflows on the internet suggest
  // using useCallback when debouncing, though I don't know why.
  // So if this breaks, maybe consider that:
  // https://stackoverflow.com/questions/61785903/problems-with-debounce-in-useeffect
  const debouncedGetFolds = useCallback(
    debounce((_filter: string | null) => {
      getFolds(_filter, null, pageNum, PAGE_SIZE).then(
        (f) => {
          setFolds(f);
          setSearchIsStale(false);
        },
        (e) => {
          props.setErrorText(e.toString());
        }
      );
    }, 300),
    [pageNum]
  );

  useEffect(() => {
    if (authenticationService.currentJwtStringValue) {
      debouncedGetFolds(filter);
    }
  }, [filter, props, debouncedGetFolds]);

  const updateSearchTerm = (e: FormEvent<HTMLFormElement> | null) => {
    if (e) {
      e.preventDefault();
    }
    setPage(1);
    setQueryStringValue("filter", filterFormValue);
    setFilter([filterFormValue]);
  };

  const updateSearchBarText = (newFilterFormValue: string) => {
    setFilterFormValue(newFilterFormValue);
    setSearchIsStale(true);
  };

  const searchForNewTerm = (newTerm: string) => {
    setPage(1);
    setFilterFormValue(newTerm);
    setQueryStringValue("filter", newTerm);
    setFilter([newTerm]);
  };

  return authenticationService.currentJwtStringValue ? (
    <div style={{ flexGrow: 1, overflowY: "scroll" }}>
      {/* <hr className="uk-divider-icon" /> */}
      <div
        className="uk-button-group uk-width-1-1 uk-padding"
        uk-tooltip="Click the magnifying glass or hit enter to execute your search."
      >
        <form
          className="uk-search uk-search-default uk-width-4-5"
          onSubmit={(e) => updateSearchTerm(e)}
        >
          <button className="uk-search-icon-flip" uk-search-icon={1}></button>
          <input
            className="uk-search-input"
            type="search"
            placeholder="Search"
            value={filterFormValue}
            onChange={(e) => updateSearchBarText(e.target.value)}
          />
        </form>
        &nbsp;
        <div className="uk-width-1-5">
          <Link to={"/newFold"}>
            <button className="uk-button uk-button-default uk-width-1-1">
              New
            </button>
          </Link>
        </div>
      </div>

      {folds ? (
        <div
          key="loadedDiv"
          style={{ opacity: searchIsStale ? "60%" : "100%" }}
        >
          {makeFoldTable(folds)}
          {folds.length === 0 ? (
            <div className="uk-flex uk-flex-center uk-flex-middle uk-margin-top">
              <div
                className="uk-button uk-button-default"
                onClick={() => {
                  searchForNewTerm("");
                }}
              >
                No folds found. View all folds?
              </div>
            </div>
          ) : null}
          <ul className="uk-pagination">
            {pageNum > 1 ? (
              <li>
                <span
                  onClick={() => setPage(pageNum - 1)}
                  style={{ userSelect: "none" }}
                >
                  <span
                    className="uk-margin-small-right"
                    uk-pagination-previous={1}
                  ></span>
                  Previous
                </span>
              </li>
            ) : null}
            {folds.length === PAGE_SIZE ? (
              <li className="uk-margin-auto-left">
                <span
                  onClick={() => setPage(pageNum + 1)}
                  style={{ userSelect: "none" }}
                >
                  Next
                  <span
                    className="uk-margin-small-left"
                    uk-pagination-next={1}
                  ></span>
                </span>
              </li>
            ) : null}
          </ul>
        </div>
      ) : (
        <div className="uk-text-center" key="unloadedDiv">
          {/* We're setting key so that the table doesn't spin... */}
          <div uk-spinner="ratio: 4" key="spinner"></div>
        </div>
      )}
    </div>
  ) : null;
}

function DashboardView(props: {
  setErrorText: (a: string) => void;
  decodedToken: DecodedJwt | null;
}) {
  if (!props.decodedToken) {
    return null; // <div uk-spinner="ratio: 4"></div>;
  }
  return (
    <AuthenticatedDashboardView
      setErrorText={props.setErrorText}
      decodedToken={props.decodedToken}
    />
  );
}

export default DashboardView;
