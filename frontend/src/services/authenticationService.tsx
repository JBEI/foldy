import { BehaviorSubject } from 'rxjs';

export interface DecodedJwt {
  user_claims: {
    email: string;
    name: string;
    type: string | null;
  };
}

export const currentJwtStringSubject = new BehaviorSubject<string | null>(
  localStorage.getItem('currentJwtString')
);

export const authenticationService = {
  logout: () => {
    localStorage.removeItem('currentJwtString');
    currentJwtStringSubject.next(null);
    window.location.href = `${import.meta.env.VITE_BACKEND_URL}/api/logout`;
  },
  get currentJwtStringValue(): string {
    return currentJwtStringSubject.value || '';
  },
};

export function redirectToLogin(): void {
  const frontendUrl = encodeURIComponent(window.location.href);
  window.open(
    `${import.meta.env.VITE_BACKEND_URL}/api/login?frontend_url=${frontendUrl}`,
    '_self'
  );
}

export function redirectToLogout(): void {
  window.open(`${import.meta.env.VITE_BACKEND_URL}/api/logout`, '_self');
}

export function getDescriptionOfUserType(userType: string): string {
  if (userType === 'admin') {
    return `Your account has "admin" permissions to ${import.meta.env.VITE_INSTITUTION} Foldy. You have edit access, plus access to debug tools available in the toolbar.`;
  } else if (userType === 'editor') {
    return `Your account has "editor" permissions to ${import.meta.env.VITE_INSTITUTION} Foldy which means you can view any structure or submit your own.`;
  } else if (userType === 'viewer') {
    return `Your account has "viewer" permissions to ${import.meta.env.VITE_INSTITUTION} Foldy.`;
  } else {
    console.error(`Unknown user type: ${userType}`);
    return '';
  }
}

export function isFullDecodedJwt(obj: any): obj is DecodedJwt {
  return (
    obj &&
    obj.user_claims &&
    typeof obj.user_claims.email === 'string' &&
    typeof obj.user_claims.name === 'string'
  );
}