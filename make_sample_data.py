"""
make_sample_data.py
-------------------
Generates a SMALL PRACTICE dataset (200 unique rows) so you can test the
training pipeline BEFORE downloading the real SMS Spam Collection.

WARNING: This is FAKE data written for practice only.
    It is NOT the real dataset. The numbers you get from training
    on this file will NOT be meaningful.
    Always use the real dataset for your final project.

How to run:
    python make_sample_data.py
Output:
    data/sample_sms.csv   (tab-separated, same format as UCI file)
"""

import os
import random

random.seed(42)

# Large pool of unique spam messages
SPAM_MESSAGES = [
    "WINNER!! You have been selected to receive a 1000 pound prize. Call NOW to claim!",
    "FREE entry to our weekly competition! Text WIN to 80085 now!",
    "Congratulations! Your mobile number has won you a cash prize of 500 dollars.",
    "Urgent! Your account will be suspended. Click here to verify now.",
    "You have 1 new voicemail. Call 09061743810 to retrieve it.",
    "URGENT: Your loan application is approved. Get 5000 pounds today!",
    "Hot singles in your area want to chat. Reply STOP to cancel.",
    "Your ringtone awaits! Download now at www.freeringtones.com",
    "Final notice: You owe 200 pounds. Pay now or face legal action.",
    "Claim your free iPhone today! Limited offer expires in 24 hours.",
    "SIX chances to win CASH! Text SIX to 87121. Cost 1.50 per msg.",
    "You are selected for a bonus reward. Reply YES to claim.",
    "FREE MESSAGE: You have won a DVD player. Call 09064019788.",
    "Exclusive deal just for you! Get 70% off. Shop now!",
    "Your credit card is compromised. Verify at http://secure-bank.link",
    "FREE gift waiting! Call 08000930705 to claim your prize now.",
    "You have been pre-approved for a cash loan of up to 10000 pounds.",
    "Congratulations! You are our lucky winner this week. Claim prize now.",
    "Double your money this week! Text INVEST to 66677 for details.",
    "IMPORTANT: Your parcel could not be delivered. Click to reschedule.",
    "WIN a brand new car! Enter now at prizecar.com. Free to enter!",
    "You have earned a reward. Text CLAIM to 85023 to receive it.",
    "Alert: Suspicious login to your account. Verify now or be locked out.",
    "Get out of debt fast! We can write off up to 80% of what you owe.",
    "Congratulations from Sky Digital. Call free on 08000407165.",
    "Last chance! Your prize voucher expires at midnight. Call 09066362231.",
    "Send a free ringtone to a friend! Text RING to 80082.",
    "Your PIN is required. Please call 0800 000 0000 to verify your details.",
    "We tried to deliver your package. Rearrange delivery at parcel-uk.com.",
    "You are entitled to compensation for a recent accident. Call 08000 000.",
]

# Large pool of unique ham messages
HAM_MESSAGES = [
    "Hey, are you coming to the party tonight?",
    "I'll be home by 8. Can you start dinner?",
    "Did you finish the assignment? I need help with question 3.",
    "Call me when you get a chance. Nothing urgent.",
    "Happy birthday! Hope you have a wonderful day.",
    "Can we reschedule the meeting to Thursday?",
    "I'm at the library. Join me when you're free.",
    "The match starts at 7 pm. Don't be late!",
    "Mum says lunch is ready. Come home soon.",
    "I forgot my umbrella. Can you bring it to college?",
    "Got the results. I passed! See you at the canteen.",
    "Are you free this weekend? Let's go for a movie.",
    "The professor postponed the class to 11 AM.",
    "Thanks for the notes. You're a lifesaver!",
    "Just landed. Will reach home in 30 minutes.",
    "Do you have the chemistry textbook with you?",
    "Reminder: submit your project report by Friday.",
    "Good morning! Have a great day at college.",
    "I'm running 10 minutes late. Save me a seat.",
    "Let's study together at the library after lunch.",
    "Can you pick up some milk on your way home?",
    "The seminar is in Room 204 today, not Room 101.",
    "Did you see the cricket score? We won!",
    "I'll send you the notes once I get home.",
    "My phone battery is almost dead. See you at 5.",
    "Don't forget we have a group meeting tomorrow at 3.",
    "The power went out at home. Using neighbour's wifi.",
    "Please call me back when you are free.",
    "Dinner is ready. Come downstairs.",
    "I submitted the form. You should get a confirmation email.",
    "The bus is running late. I'll be there in 20 minutes.",
    "Can you check if the lab is open today?",
    "Just finished my exam. It went really well!",
    "Bring an extra pen tomorrow. Mine broke.",
    "Mom called. She wants to know if you are coming home this weekend.",
    "I left my bag in the classroom. Have you seen it?",
    "The wifi password is spamguard2024.",
    "We have a test next Monday on data structures.",
    "I booked the tickets. We're going on Saturday!",
    "Please don't be late. The interview starts at 10 sharp.",
    "Have you eaten yet? Let's grab something from the canteen.",
    "I need to borrow your notes from last week's lecture.",
    "Sir postponed the viva to next week.",
    "Can you send me the project file? I forgot to copy it.",
    "The lab practicals are from 2 to 5 today.",
    "Happy Diwali! Hope you have a safe and joyful celebration.",
    "My sister is getting married next month. You're invited!",
    "Let me know when you reach the station.",
    "Did you complete the mini project? We need to submit it tomorrow.",
    "Good luck for your exam. You'll do great!",
    "The college fest is next week. Are you participating?",
    "See you at the bus stop at 8 am.",
    "I left your charger on the table. Come get it.",
    "Our group project topic was approved!",
    "Which topic did you choose for the seminar?",
    "The cafeteria is closed today. Bring your own food.",
    "Please forward the assignment to everyone in the group.",
    "We have extra classes this Saturday for the backlog.",
    "I'll pick you up at 9. Be ready.",
    "The results will be announced on Friday. Fingers crossed!",
    "I'm in the canteen. Join us when you're done.",
    "Your parcel arrived. I kept it inside the house.",
    "Don't forget to bring your ID card for the exam.",
    "We passed! Let's celebrate after college.",
    "The group photo is at 4 pm near the main gate.",
    "Chapter 5 and 6 are important for tomorrow's test.",
    "I'm sorry I missed your call. What's up?",
    "Can you help me with the Python code tonight?",
    "We need at least 75% attendance to write the exam.",
    "The electricity bill is due tomorrow. Please pay it.",
    "My train is at 6 pm. Can someone drop me at the station?",
    "Are you joining the machine learning elective next semester?",
    "The placement drive is next Tuesday. Please register soon.",
    "I got selected for the internship! Thank you for your help.",
    "Library books are due this Friday. Please return them.",
    "The auditorium is booked for our farewell on the 20th.",
    "The HOD wants the project synopsis by end of day.",
    "Which stream did you choose? I went with AI and ML.",
    "I'll explain the doubt during lunch break.",
    "We are doing a presentation on spam detection for our mini project.",
    "Let me know if you need the reference book.",
    "The viva questions were mostly on the confusion matrix and F1 score.",
    "Can you send me the meeting link? I can't find the email.",
    "We have a hackathon this weekend. Want to team up?",
    "Just saw your message. Call me when free.",
    "I'll finish this by tonight and share it in the group.",
]

# Build dataset with approx 87% ham, 13% spam, all unique
rows = []
for msg in HAM_MESSAGES:
    rows.append(("ham", msg))
for msg in SPAM_MESSAGES:
    rows.append(("spam", msg))

random.shuffle(rows)

os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "sample_sms.csv")

with open(output_path, "w", encoding="utf-8") as f:
    for label, message in rows:
        safe_message = message.replace("\t", " ")
        f.write(f"{label}\t{safe_message}\n")

ham_count  = sum(1 for r in rows if r[0] == "ham")
spam_count = sum(1 for r in rows if r[0] == "spam")

print(f"[OK] Practice dataset written to: {output_path}")
print(f"     Total rows : {len(rows)}")
print(f"     Ham (legit): {ham_count}")
print(f"     Spam       : {spam_count}")
print()
print("[NOTE] This is FAKE data for testing only. Use the real dataset for your project.")
print("       Real dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection")
