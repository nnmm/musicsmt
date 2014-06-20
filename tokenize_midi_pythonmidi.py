import midi

def events_to_dict(chan):
    """Filters out non-note events from the event list, converts it to a list of dicts and sorts it"""
    allNoteEvents = (e for e in chan if hasattr(e, "pitch"))
    unmatchedNoteOnEvents = {}
    # queue contains notes whose start >= start of some of the notes in unmatchedNoteOnEvents
    queue = []
    for e in allNoteEvents:
        if e.name == 'Note On' and e.velocity != 0:
            # if e.pitch in unmatchedNoteOnEvents.keys(): print("Note On twice")
            unmatchedNoteOnEvents[e.pitch] = {"pitch": e.pitch, "velocity": e.velocity, "start": e.tick}
        else:
            # Find matching note
            if e.pitch in unmatchedNoteOnEvents.keys():
                note = unmatchedNoteOnEvents[e.pitch]
                del unmatchedNoteOnEvents[e.pitch]
                note["end"] = e.tick
                # if start time is strictly lower than everything (else) in unmatchedNoteOnEvents, first yield
                # everything with lower start from queue and self
                # else, append to queue
            else:
                # it happens, just ignore
                # print("Note Off before Note On")
                pass


def notes_to_token(notes):
    strings = []
    for note in notes:
        strings.append("p"+str(note["pitch"]) + "d" + str(note["end"]-note["start"]))
    return ''.join(strings)


def relativize(notes):
    # first item
    prev = next(notes)
    prev["pitch_rel"] = 0
    yield prev
    for n in notes:
        n["pitch_rel"] = n["pitch"] - prev["pitch"]
        yield n
        prev = n


def main():
    example_file = '/home/internet/Dropbox/Bachelorarbeit/Daten/canon.mid'
    pattern = midi.read_midifile(example_file)
    pattern.make_ticks_abs()
    # sort by start
    # relative pitch
    prevstart = -1
    for n in relativize(events_to_dict(pattern[1])):
        if prevstart == n["start"]:
            print("!"),
        print(n)
        prevstart = n["start"]


if __name__ == "__main__":
    main()