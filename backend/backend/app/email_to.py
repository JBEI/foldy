"""Copied from the email_to package, for adding ssl support."""
from __future__ import absolute_import

# -*- coding: utf-8 -*-

"""Main module."""
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib

import markdown
import premailer


class Message(object):
    """A simple email message made of Markdown strings

    A Message is a simplified version of a MIMEMultipart email message.
    Instead of having to seperately construct HTML and text version of
    an email, you can build a Message by adding lines of Markdown formatted
    strings.

    args:
        body (str, bytes, iterable of strings): The body of the email.
            It does not need to be complete, as you can add to it later.
        style (str): CSS formatting to be applied to HTML formatted verion
            of email.
        server (EmailServer): An EmailServer instance allowing a message
            to be sent directly from it's `send` method.
    """

    def __init__(self, body=None, style=None, server=None):
        if body is None:
            self.body = []
        else:
            if any((isinstance(body, str), isinstance(body, bytes))):
                body = [body]
            self.body = body
        self.style = style
        self.server = server

    def add(self, line):
        """Adds a new Markdown formatted line to the current email body"""
        self.body.append(line)

    @property
    def html(self):
        """Returns HTML formatted and styled version of body"""
        html = markdown.markdown("\n".join(self.body))
        if self.style:
            return premailer.transform("<style>\n" + self.style + "\n</style>\n" + html)
        return html

    def __str__(self):
        return "\n".join(self.body)

    def __repr__(self):
        return "<Message: {0} and {1} more lines>".format(
            self.body[0], (len(self.body) - 1)
        )

    def mime(self):
        """Returns a MIMEMultipart message"""
        msg = MIMEMultipart("alternative")

        msg.attach(MIMEText(str(self), "plain"))
        msg.attach(MIMEText(self.html, "html"))

        return msg

    def send(self, send_to, subject):
        """Sends the formatted message to given recripient

        args:
            send_to (:obj:`str`, iterable of :obj:`str`): Email addresses to
                send to
            subject (str): Subject line of email to send

        """
        self.server.send_message(self, send_to, subject)


class EmailServer(object):
    """Connection to a specific email server

    args:
        url (str): URL for the SMTP server
        port (int): SMTP port
        email_address (str): Email address to log into server and send from
        password (str): Password for email address on SMTP server
    """

    def __init__(self, url, port, email, password):
        self.url = url
        self.port = port
        self.email_address = email
        self.password = password
        self.server = None

    def _login(self):
        # self.server = smtplib.SMTP_SSL(self.url, self.port)
        self.server = smtplib.SMTP(self.url, self.port)
        self.server.starttls()
        self.server.login(self.email_address, self.password)

    def _logout(self):
        self.server.quit()

    def quick_email(self, send_to, subject, body, style=None):
        """Compose and send an email in a single call

        args:
            send_to (str):
            subject (str):
            body (:obj:`str`, iterable of :obj:`str`): Markdown formatted
                string or iterable of strings composing the message body
            style (str): CSS formatting for the message
        """
        message = Message(body, style=style)

        self.send_message(message, send_to, subject)

    def send_message(self, message, send_to, subject):
        """Send a precomposed Message

        args:
            message (Message): Completed message to send
            send_to (str): Email address to send the message to
            subject (str): Subject of the email
        """
        message = message.mime()

        message["From"] = self.email_address
        message["To"] = send_to

        message["Subject"] = subject

        self._login()
        self.server.sendmail(self.email_address, send_to, message.as_string())
        self._logout()

    def message(self, body=None, style=None):
        """Returns a Message object

        args:
            body (str, bytes, iterable of strings): The body of the email.
                It does not need to be complete, as you can add to it later.
            style (str): CSS formatting to be applied to HTML formatted verion
                of email.
        """
        return Message(body=body, style=style, server=self)
