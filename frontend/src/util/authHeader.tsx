import { authenticationService } from "../services/authentication.service";

export function authHeader(): Record<string, string> {
    const currentJwtString = authenticationService.currentJwtStringValue;
    return currentJwtString
        ? { Authorization: `Bearer ${currentJwtString}` }
        : {};
}

export function jsonBodyAuthHeader(): Record<string, string> {
    return {
        ...authHeader(),
        "Content-Type": "application/json",
    };
}