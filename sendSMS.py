from twilio.rest import Client

def send(message):
    # Twilio credentials (replace with your actual credentials)
    account_sid = 'AC060e166b40f8443e8d57ecf9095b0f94'
    auth_token = '7a654a6086776183ff9d7ad124c01ae8'
    client = Client(account_sid, auth_token)

    # Replace with your Twilio phone number and recipient's number
    message = client.messages.create(
        body=message,
        to='+919597830040', #fill in appropriately
        from_='+13159037297'  # Recipient's phone number

    )

    print("SMS sent: ", message.sid)
