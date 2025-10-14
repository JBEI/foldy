import React, { ReactNode, CSSProperties } from 'react';

interface DataTableContainerProps {
    children: ReactNode;
    style?: CSSProperties;
}

export const DataTableContainer: React.FC<DataTableContainerProps> = ({
    children,
    style = {}
}) => {
    const defaultStyle: CSSProperties = {
        width: "auto",
        height: "auto",
        marginTop: "20px",
        ...style
    };

    return <div style={defaultStyle}>{children}</div>;
};
