"""MkDocs hooks: rewrite the docs' links that point outside docs/ (the lab log, exp/out figures, source trees) to
their site or GitHub locations, so the pages read the same on GitHub and on the built site."""
import re

TREE = "https://github.com/2imi9/olmoearth_inferenceX-record/tree/main/"
PACKAGE_TREE = "https://github.com/2imi9/olmoearth_inferenceX/tree/main/"   # oe_inferencex/ is in the code repository


def on_page_markdown(markdown, page, config, files):
    up = "../" * page.file.src_uri.count("/")
    markdown = markdown.replace("](../../exp/NOTES.md)", f"]({up}LabLog.md)").replace("](../exp/NOTES.md)", f"]({up}LabLog.md)")
    markdown = re.sub(r"\]\((?:\.\./)+exp/out/([^)]+\.png)\)", lambda m: f"]({up}exp_out/{m.group(1)})", markdown)
    markdown = re.sub(r"\]\((?:\.\./)+(exp|oe_inferencex|tests|scripts)/\)", lambda m: f"]({PACKAGE_TREE if m.group(1) == 'oe_inferencex' else TREE}{m.group(1)})", markdown)
    return markdown
