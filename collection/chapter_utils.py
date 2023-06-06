import re

def parse_timestamp(description):
    timestamp_lines = []

    lines = description.split("\n")
    for i, line in enumerate(lines):
        if re.search("\d{1}:\d{2}", line):
            line = re.sub(r'http\S+', '', line)     # remove all http urls
            timestamp_lines.append(line)
    return timestamp_lines

def extract_timestamp(s):
    if re.search("\d{3}:", s) or re.search(":\d{3}", s) or re.search("\d{2}:\d{2}:\d{2}:\d{2}", s):  # buggy like https://www.youtube.com/watch?v=vBv5G5emgl0&ab_channel=ScreenRant or https://www.youtube.com/watch?v=HucIsg_JhX0&ab_channel=PeriscopeFilm
        return "", -1, -1, -1
    r = re.search("\d{2}:\d{2}:\d{2}", s)
    if r:
        si, ei = r.regs[0]
    else:
        r = re.search("\d{1}:\d{2}:\d{2}", s)
        if r:
            si, ei = r.regs[0]
        else:
            r = re.search("\d{2}:\d{2}", s)
            if r:
                si, ei = r.regs[0]
            else:
                r = re.search("\d{1}:\d{2}", s)
                if r:
                    si, ei = r.regs[0]
                else:
                    return "", -1, -1, -1

    timestamp = s[si:ei]
    ts = timestamp.split(":")
    ts.reverse()
    sec = 0
    for i in range(len(ts)):
        if i == 0:
            sec += int(ts[i])
        elif i == 1:
            sec += int(ts[i]) * 60
        elif i == 2:
            sec += int(ts[i]) * 3600

    return s[si:ei], sec, si, ei

def clean_str(s):
    """
    Remove all special char at the beginning and the end.
    Use to clean chapter title string
    """
    start_idx = 0
    for i in range(len(s)):
        if s[i].isalnum():
            start_idx = i
            break

    end_idx = len(s)
    for i in reversed(range(len(s))):
        if s[i].isalnum():
            end_idx = i + 1
            break

    if all(not s[i].isalnum() for i in range(len(s))):
        return ''
    return s[start_idx: end_idx]