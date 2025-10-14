import React, { useEffect } from "react";
import { BehaviorSubject } from "rxjs";
import { map } from "rxjs/operators";
import { Button, Space, Typography, Dropdown, Avatar } from 'antd';
import { UserOutlined, LogoutOutlined } from '@ant-design/icons';

const { Text } = Typography;

// Based on:
// https://jasonwatmore.com/post/2019/04/06/react-jwt-authentication-tutorial-example

export interface UserAttributes {
    beta_access: boolean | null;
}
export interface DecodedJwt {
    user_claims: {
        email: string;
        name: string;
        type: string | null;
        attributes: UserAttributes | null;
    };
}

export function getDescriptionOfUserType(userType: string) {
    if (userType === "admin") {
        return `Your account has "admin" permissions to ${import.meta.env.VITE_INSTITUTION} Foldy. You have edit access, plus access to debug tools available in the toolbar. The RQ page shows the status of the Redis Queue, which manages jobs. The DBs page allows direct edit access to all underlying databases, built on Flask-Admin. The Sudo Page contains some convenient buttons for manipulating folds.`;
    } else if (userType === "editor") {
        return `Your account has "editor" permissions to ${import.meta.env.VITE_INSTITUTION} Foldy which means you can view any structure or submit your own. Check out the instructions in the About page for details.`;
    } else if (userType === "viewer") {
        return `Your account has "viewer" permissions to ${import.meta.env.VITE_INSTITUTION} Foldy which means you can view any public folds and associated data, but cannot submit your own compute jobs. See the Foldy manuscript and codebase to set up a Foldy instance at your institution!`;
    } else {
        console.error(`Unknown user type ${userType}`);
        return "";
    }
}

export function isFullDecodedJwt(obj: any): obj is DecodedJwt {
    return (
        obj &&
        obj.user_claims &&
        typeof obj.user_claims.email === "string" &&
        typeof obj.user_claims.name === "string"
    );
}

export const currentJwtStringSubject: BehaviorSubject<string | null> = (() => {
    return new BehaviorSubject(localStorage.getItem("currentJwtString"));
})();

export const authenticationService = {
    logout,
    currentJwtString: currentJwtStringSubject.asObservable().pipe(
        map((v: string | null): string => {
            return v || "";
        })
    ),
    get currentJwtStringValue() {
        return currentJwtStringSubject.value || "";
    },
};

export function redirectToLogin() {
    const frontend_url = encodeURIComponent(window.location.href);
    window.open(
        `${import.meta.env.VITE_BACKEND_URL}/api/login?frontend_url=${frontend_url}`,
        "_self"
    );
}

export function redirectToLogout() {
    window.open(`${import.meta.env.VITE_BACKEND_URL}/api/logout`, "_self");
}

function logout() {
    // remove user from local storage to log user out
    localStorage.removeItem("currentJwtString");
    currentJwtStringSubject.next(null);

    redirectToLogout();
}

export function LoginButton(props: {
    setToken: (a: string) => void;
    decodedToken: DecodedJwt | null;
    isExpired: boolean;
}) {
    useEffect(() => {
        authenticationService.currentJwtString.subscribe(props.setToken);
    }, [props.setToken]);

    if (props.decodedToken && !props.isExpired) {
        return (
            <Space size="middle" style={{ color: 'inherit' }}>
                <Text
                    style={{ color: 'inherit' }}
                    className="uk-visible@m"
                >
                    Logged in as {props.decodedToken.user_claims.name}{props.decodedToken.user_claims.attributes?.beta_access ? ' (BETA ACCESS)' : null}.
                </Text>
                <Button
                    onClick={(e) => {
                        e.preventDefault();
                        authenticationService.logout();
                    }}
                >
                    Logout
                </Button>
            </Space>
        );
    } else {
        return (
            <Button
                onClick={(e) => {
                    e.preventDefault();
                    redirectToLogin();
                }}
                style={{ backgroundColor: '#0000', color: 'white' }}
            >
                Login
            </Button>
        );
    }
}

export function UserProfileDropdown(props: {
    setToken: (a: string) => void;
    decodedToken: DecodedJwt | null;
    isExpired: boolean;
}) {
    useEffect(() => {
        authenticationService.currentJwtString.subscribe(props.setToken);
    }, [props.setToken]);

    if (props.decodedToken && !props.isExpired) {
        const menuItems = [
            {
                key: 'user-info',
                label: (
                    <div style={{ padding: '4px 0' }}>
                        <Text strong>{props.decodedToken.user_claims.name}</Text>
                        {props.decodedToken.user_claims.attributes?.beta_access && (
                            <div style={{ fontSize: '12px', color: '#1890ff', fontWeight: 'bold' }}>
                                BETA ACCESS
                            </div>
                        )}
                        <div style={{ fontSize: '12px', color: '#666', marginTop: '2px' }}>
                            {props.decodedToken.user_claims.email}
                        </div>
                    </div>
                ),
                disabled: true,
            },
            {
                type: 'divider' as const,
            },
            {
                key: 'logout',
                label: 'Logout',
                icon: <LogoutOutlined />,
                onClick: () => {
                    authenticationService.logout();
                },
            },
        ];

        return (
            <Dropdown
                menu={{ items: menuItems }}
                placement="bottomRight"
                arrow
                trigger={['click']}
            >
                <Avatar
                    size="small"
                    icon={<UserOutlined />}
                    style={{
                        backgroundColor: 'rgba(255, 255, 255, 0.2)',
                        color: 'white',
                        cursor: 'pointer',
                        border: '1px solid rgba(255, 255, 255, 0.3)'
                    }}
                />
            </Dropdown>
        );
    } else {
        return (
            <Button
                onClick={(e) => {
                    e.preventDefault();
                    redirectToLogin();
                }}
                style={{ backgroundColor: '#0000', color: 'white' }}
            >
                Login
            </Button>
        );
    }
}
