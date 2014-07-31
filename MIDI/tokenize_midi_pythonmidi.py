#!/usr/bin/python
# coding=utf-8
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
            # You could also add e.velocity
            unmatchedNoteOnEvents[e.pitch] = (e.tick, e.pitch)
        else:
            # Find matching note to this Note Off event
            if e.pitch in unmatchedNoteOnEvents.keys():
                note = unmatchedNoteOnEvents[e.pitch]
                del unmatchedNoteOnEvents[e.pitch]
                # Add the duration
                note = note + (e.tick - note[0],)
                queue.append(note)
                queue.sort()
                # If we have two simultaneous events, they will both go into unmatchedNoteOnEvents.
                # Then, there will be matching Note Off events in some order.
                # If the first one arrives, in will be added to the queue.
                # Now we look if – by removing it from unmatchedNoteOnEvents – there are notes in queue
                # that we can yield, i.e. whose start is < that of the earliest note in unmatchedNoteOnEvents.
                # < vs <= matters if we have simultaneous notes: we want to wait until they are both in queue
                # so queue.sort() sorts them in order of ascending pitch
                unmatchedTuplesSorted = sorted(unmatchedNoteOnEvents.values())
                if unmatchedTuplesSorted:
                    earliest_unmatched_start = unmatchedTuplesSorted[0][0]
                    cutoff_index = 0
                    for i, n in enumerate(queue):
                        if n[0] < earliest_unmatched_start:
                            # There is no earlier note, yield
                            yield n
                            cutoff_index = cutoff_index + 1
                    queue = queue[cutoff_index:]
                else:
                    # No unmatchedNoteOnEvents, so we can just yield the whole (sorted) queue
                    for n in queue:
                        yield n
                    queue = []
            else:
                # it happens, just ignore
                # print("Note Off before Note On")
                pass



def notes_to_token(notes):
    strings = []
    for note in notes:
        strings.append("p" + str(note[3]) + "d" + str(note[2]))
    return ''.join(strings)


def relativize(notes):
    # first item
    prev = next(notes)
    prev = prev + (0,)
    yield prev
    for n in notes:
        n = n + (n[1] - prev[1],)
        yield n
        prev = n


def main():
    example_file = '/home/internet/Dropbox/Bachelorarbeit/Daten/canon.mid'
    pattern = midi.read_midifile(example_file)
    pattern.make_ticks_abs()

    notes = list(relativize(events_to_dict(pattern[2])))
    i = 2
    result = []
    while i < len(notes):
        result.append(notes_to_token(notes[i-2:i]))
        i = i + 2
    f = open('musictokens.txt', 'w')
    f.write("\n".join(result))
    f.close()


if __name__ == "__main__":
    main()