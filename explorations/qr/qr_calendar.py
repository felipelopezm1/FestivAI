# create_calendar_file.py

events = [
    {
        "summary": "Martin Garrix Live",
        "start": "20250701T200000Z",
        "end": "20250701T220000Z",
        "location": "Sunset Stage",
        "description": "Live set by Martin Garrix"
    },
    {
        "summary": "Remi Wolf Set",
        "start": "20250701T223000Z",
        "end": "20250701T233000Z",
        "location": "Garden Stage",
        "description": "Funky & upbeat performance by Remi Wolf"
    },
    {
        "summary": "Wunderhorse Show",
        "start": "20250702T170000Z",
        "end": "20250702T180000Z",
        "location": "River Stage",
        "description": "Alternative rock vibes with Wunderhorse"
    }
]

calendar_content = "BEGIN:VCALENDAR\nVERSION:2.0\n"

for event in events:
    calendar_content += f"""BEGIN:VEVENT
SUMMARY:{event['summary']}
DTSTART:{event['start']}
DTEND:{event['end']}
LOCATION:{event['location']}
DESCRIPTION:{event['description']}
END:VEVENT
"""

calendar_content += "END:VCALENDAR"

with open("festival_calendar.ics", "w") as f:
    f.write(calendar_content)

print("festival_calendar.ics created successfully!")
