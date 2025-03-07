import React, { useState } from "react";

interface FoldyMascotProps {
  text: string;
  moveTextAbove: boolean;
}

export function FoldyMascot(props: FoldyMascotProps) {
  const [hidden, setHidden] = useState<boolean>(false);
  if (hidden) {
    return null;
  }
  return (
    <div>
      <div
        style={{
          position: "fixed",
          bottom: props.moveTextAbove ? "262px" : "210px",
          right: props.moveTextAbove ? "34px" : "180px",
        }}
        className={
          props.moveTextAbove ? "sbbox sbtriangleabove" : "sbbox sbtriangle"
        }
      >
        {props.text}
        <div
          style={{
            position: "absolute",
            top: "0px",
            right: "8px",
          }}
          onClick={() => setHidden(!hidden)}
        >
          X
        </div>
      </div>
      <img
        style={{
          width: "250px",
          position: "fixed",
          bottom: "10px",
          right: "10px",
          zIndex: 1,
        }}
        src={`/pksito.gif`}
        alt=""
      />
    </div>
  );
}
