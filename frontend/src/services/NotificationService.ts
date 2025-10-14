import UIkit from 'uikit';
import './NotificationService.scss'

type NotificationType = 'success' | 'warning' | 'danger' | 'primary';

interface NotificationOptions {
    message: string;
    type?: NotificationType;
    timeout?: number;
    position?: 'top-left' | 'top-center' | 'top-right' | 'bottom-left' | 'bottom-center' | 'bottom-right';
}

class NotificationService {
    private static instance: NotificationService;

    private constructor() {}

    static getInstance(): NotificationService {
        if (!NotificationService.instance) {
            NotificationService.instance = new NotificationService();
        }
        return NotificationService.instance;
    }

    show({
        message,
        type = 'primary',
        timeout = 3000,
        position = 'top-right'
    }: NotificationOptions): void {
        UIkit.notification({
            message,
            timeout,
            pos: position,
            status: type
        })
        // UIkit.notification({
        //     message,
        //     status: type,
        //     timeout,
        //     pos: position,
        //     container: '.uk-notification-container'
        // });
    }

    success(message: string): void {
        this.show({ message, type: 'success' });
    }

    warning(message: string): void {
        this.show({ message, type: 'warning' });
    }

    error(message: string): void {
        this.show({ message, type: 'danger', timeout: 5000 });
    }

    info(message: string): void {
        this.show({ message, type: 'primary' });
    }
}

export const notify = NotificationService.getInstance();
