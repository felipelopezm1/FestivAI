import qrcode

# Github pages link where qr code is nested
github_calendar_url = "https://git.arts.ac.uk/pages/23041393/PML_Project_Group_1/festival_calendar.ics"


# Generate the QR code
qr = qrcode.QRCode(
    version=1,
    box_size=10,
    border=4,
)
qr.add_data(github_calendar_url)
qr.make(fit=True)

# Show qr code
img = qr.make_image(fill_color="black", back_color="white")
img.show()

# If we wanna save the pngs
img.save("calendar_qr.png")

print(f"QR code generated for: {github_calendar_url}")

