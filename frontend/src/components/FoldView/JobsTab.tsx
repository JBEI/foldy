import React from "react";
import { Invokation } from "../../services/backend.service";

interface JobsTabProps {
    jobs: Invokation[] | null;
}

const JobsTab: React.FC<JobsTabProps> = ({ jobs }) => {
    const formatStartTime = (jobstarttime: string | null) => {
        return jobstarttime
            ? new Date(jobstarttime).toLocaleString("en-US", {
                timeStyle: "short",
                dateStyle: "short",
            })
            : "Not Started / Unknown";
    };

    const formatRunTime = (jobRunTime: number | null) => {
        return jobRunTime
            ? `${Math.floor(jobRunTime / (60 * 60))} hr ${Math.floor(jobRunTime / 60) % 60
            } min ${Math.floor(jobRunTime) % 60} sec`
            : "NA";
    };

    if (!jobs) return null;

    return (
        <span>
            <h2>Invokations</h2>
            <div style={{ display: "flex", flexDirection: "row" }}>
                <div style={{ overflowX: "scroll", flexGrow: 1 }}>
                    <table className="uk-table uk-table-hover uk-table-striped uk-table-small">
                        <thead>
                            <tr>
                                <th>Type</th>
                                <th className="uk-text-nowrap">State</th>
                                <th className="uk-text-nowrap">Start time</th>
                                <th className="uk-text-nowrap">Runtime</th>
                                <th className="uk-text-nowrap">Logs</th>
                            </tr>
                        </thead>
                        <tbody>
                            {[...jobs].map((job: Invokation) => (
                                <tr key={`${job.job_id}_${job.id}`}>
                                    <td className="uk-text-nowrap" uk-tooltip={job.type}>
                                        {job.type}
                                    </td>
                                    <td className="uk-text-nowrap" uk-tooltip={job.state}>
                                        {job.state}
                                    </td>
                                    <td
                                        className="uk-text-nowrap"
                                        uk-tooltip={formatStartTime(job.starttime)}
                                    >
                                        {formatStartTime(job.starttime)}
                                    </td>
                                    <td
                                        className="uk-text-nowrap"
                                        uk-tooltip={formatRunTime(job.timedelta_sec)}
                                    >
                                        {formatRunTime(job.timedelta_sec)}
                                    </td>
                                    <td className="uk-text-nowrap">
                                        <a href={`#logs_${job.id?.toString()}`}>View</a>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
            <span>
                {[...jobs].map((job: Invokation) => (
                    <div
                        id={`logs_${job.id?.toString()}`}
                        key={job.id || "jobid should not be null"}
                    >
                        <h3>{job.type} Logs</h3>
                        <pre>Command: {job.command}</pre>
                        <pre>{job.log}</pre>
                    </div>
                ))}
            </span>
        </span>
    );
};

export default JobsTab;