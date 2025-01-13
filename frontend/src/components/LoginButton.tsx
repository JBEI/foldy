import React, { useEffect } from 'react';
import { authenticationService, DecodedJwt, currentJwtStringSubject, redirectToLogin } from '../services/authentication.service';
import { map } from "rxjs/operators";

interface LoginButtonProps {
    setToken: (token: string) => void;
    decodedToken: DecodedJwt | null;
    isExpired: boolean;
}

const LoginButton: React.FC<LoginButtonProps> = ({ setToken, decodedToken, isExpired }) => {
    useEffect(() => {
        const subscription = currentJwtStringSubject.asObservable().pipe(
            map((v: string | null): string => {
                return v || "";
            })
        ).subscribe(setToken);
        return () => subscription.unsubscribe(); // Clean up subscription on unmount
    }, [setToken]);

    if (decodedToken && !isExpired) {
        return (
            <div>
                <span className="uk-visible@m">Logged in as {decodedToken.user_claims.name}.</span>
                <button
                    className="uk-button uk-button-default uk-margin-left"
                    onClick={(e) => {
                        e.preventDefault();
                        authenticationService.logout();
                    }}
                >
                    <span className="icon" style={{ color: '#fff' }}>
                        Logout
                    </span>
                </button>
            </div>
        );
    }

    return (
        <div>
            <button
                className="uk-button uk-button-default"
                onClick={(e) => {
                    e.preventDefault();
                    redirectToLogin();
                }}
            >
                <span className="icon" style={{ color: '#fff' }}>
                    Login
                </span>
            </button>
        </div>
    );
};

export default LoginButton;