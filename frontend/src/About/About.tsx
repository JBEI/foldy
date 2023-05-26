import React from "react";
import { getDescriptionOfUserType } from "./../services/authentication.service";

const About = (props: { userType: string | null }) => (
  <div data-testid="About">
    <h1>About</h1>
    <p>
      Thanks for using <b>{process.env.REACT_APP_INSTITUTION} Foldy</b>, a
      Keasling Lab project to get AlphaFold structures in the hands of more
      folks.
    </p>
    <p>
      Users at {process.env.REACT_APP_INSTITUTION} have <b>editor</b> permission
      to this Foldy instance, all others have <b>viewer</b> permissions.{" "}
      {props.userType ? getDescriptionOfUserType(props.userType) : null}
    </p>
    <p>
      From the home page of this site (accesible from the button in the top
      left), editors can submit amino acid sequences by clicking the NEW button
      next to the search bar. The name must be unique and the amino acid
      sequence must only contain canonical amino acids. We will run AlphaFold on
      your sequence, typically within the next day, and your results will be
      available for download as a PDB within this site. That's it!
    </p>
    <p>Some of our current limitations:</p>
    <ul>
      <li>sequences over 3,000 amino acids are likely to fail</li>
      <li>
        before submitting more than 100 sequences per day please reach out to
        your Foldy administrators.
      </li>
    </ul>
    <h2>Q{"&"}A</h2>
    <h4>When will my fold finish?</h4>
    We're aiming for {"<"} 1 day turnaround. If it has been more than 48 hours
    since you submitted your fold, please retry your steps from the "Action"
    tab. If that doesn't work, email your Foldy administrators.
    <h4>Why did my fold fail?</h4>
    Foldy can fail for a few reasons, including running out of memory (if the
    sequence is too long) or limited resource availability in the cloud. Check
    out the Logging tab in the Fold page for the logs from the run, and feel
    comfortable retrying the failing steps from the Action tab. If that doesn't
    work, try contacting your Foldy administrators.
    <h4>Where can I learn more about the architecture of this website?</h4>
    See the Foldy publication.
    <h4>Where do I get the rest of the AlphaFold output?</h4>
    All output files are available through the "Actions" tab in the Fold view.
    Note that at present, large files may be corrupted.
    <h4>Where do I put requests?</h4>
    Please file feature requests, bug reports, and comments on the{" "}
    <a href="https://github.com/JBEI/foldy">Foldy Github page</a>.
    <h2>Citation</h2>
    Please cite the{" "}
    <a href="https://www.biorxiv.org/content/10.1101/2023.05.11.540333v1">
      Foldy paper
    </a>
    , as well as any tools used within (such as Alphafold, Autodock Vina, PFam,
    DiffDock), when publishing.
  </div>
);

export default About;
