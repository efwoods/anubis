"""Mailbox access for the personal avatar.

``imap_client`` speaks to a mail server; ``mailbox_tools`` wraps that in the
tools the model calls. Neither is imported at package scope, because the tool
layer is built per turn and only for an avatar that actually has a mailbox
connected — importing a mail stack into every process that touches this package
would work against the repository's cold-start rule.
"""
