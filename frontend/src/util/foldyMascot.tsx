import React, { useState } from "react";

interface FoldyMascotProps {
    text: string;
    moveTextAbove: boolean;
    isCartwheeling: boolean;
    isKanKaning: boolean;
}

const spinningStyles = `
@keyframes gentlerock {
  0% {
    transform: rotate(-3deg);
    animation-timing-function: cubic-bezier(0.5, 0, 0.5, 1);
  }
  50% {
    transform: rotate(3deg);
    animation-timing-function: cubic-bezier(0.5, 0, 0.5, 1);
  }
  100% {
    transform: rotate(-3deg);
  }
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

@keyframes slide
{
    0% {
        left: -300px;
    }
    100% {
        left: 110%;
    }
}

@keyframes kankan {
    0% {
        transform: translateY(0) rotate(0deg);
    }
    25% {
        transform: translateY(-80px) rotate(10deg);
    }
    50% {
        transform: translateY(-100px) rotate(0deg);
    }
    75% {
        transform: translateY(-80px) rotate(-10deg);
    }
    100% {
        transform: translateY(0) rotate(0deg);
    }
}

@keyframes dvdBounce {
    0% {
        transform: translate(calc(-50vw), calc(-100vh + 250px));
    }
    25% {
        transform: translate(calc(-100vw + 250px), calc(-50vh));
    }
    50% {
        transform: translate(calc(-50vw), calc(-0vh));
    }
    75% {
        transform: translate(calc(-0vw), calc(-50vh));
    }
    100% {
        transform: translate(calc(-50vw), calc(-100vh + 250px));
    }
}

.gentlerocking {
  animation: gentlerock 2s infinite;
}
.cartwheeling {
  animation: spin 2s linear infinite, slide 6s linear forwards;
}
.kankaning {
  animation: kankan 1s linear infinite, slide 10s linear forwards;
}
.dvdBouncing {
  animation: dvdBounce 20s linear infinite;
}
`;

export function FoldyMascot(props: FoldyMascotProps) {
    const [hidden, setHidden] = useState<boolean>(false);
    if (hidden) {
        return null;
    }

    const classNames = [];
    if (props.isCartwheeling) {
        classNames.push("cartwheeling");
    }
    else if (props.isKanKaning) {
        classNames.push("kankaning");
    }
    else {
        classNames.push("gentlerocking");
    }

    return (
        <div>
            <style>{spinningStyles}</style>
            {
                props.isCartwheeling ? null : (
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
                )
            }
            <img
                style={{
                    width: "250px",
                    position: "fixed",
                    bottom: "10px",
                    right: "10px",
                    zIndex: 1,
                }}
                className={classNames.join(" ")}
                src={`/pksito.gif`}
                alt=""
            />
        </div>
    );
}


interface FoldingAtTheDisco {
    enabled: boolean;
}

export function StandaloneFoldyMascot() {
    return (
        <div>
            <style>{spinningStyles}</style>
            <img
                style={{
                    width: "200px",
                    display: "inline-block",
                    marginBottom: "24px",
                }}
                className={"gentlerocking"}
                src={`/pksito.gif`}
                alt="Foldy Mascot"
            />
        </div>
    );
}

export function FoldingAtTheDisco(props: FoldingAtTheDisco) {
    if (!props.enabled) {
        return null;
    }

    return (
        <div>
            <style>{spinningStyles}</style>
            <img
                style={{
                    width: "250px",
                    position: "fixed",
                    bottom: "10px",
                    right: "10px",
                    zIndex: 1,
                }}
                className={"dvdBouncing"}
                src={`/disco.gif`}
                alt=""
            />
        </div>
    );
}
