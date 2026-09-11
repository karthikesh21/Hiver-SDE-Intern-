"""
Build, curate, and validate the 200-sample Golden Evaluation Set
with strict isolation from the retrieval corpus to prevent data leakage.
"""

import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Any
from src.config import (
    GOLDEN_SET_PATH,
    HISTORICAL_RESOLUTION_PATH,
    INTENTS_PATH,
    RANDOM_SEED
)

def build_golden_evaluation_set():
    random.seed(RANDOM_SEED)
    
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        intents_def = json.load(f)
        
    # Read existing processed interactions
    historical_records = []
    with open(HISTORICAL_RESOLUTION_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                historical_records.append(json.loads(line))
                
    print(f"Total historical pool: {len(historical_records)} records.")
    
    # 200 curated, human-verified golden test examples across the 8 intents
    # Each intent has exactly 25 diverse examples reflecting real customer query styles,
    # varied message lengths, ambiguity levels, and realistic escalation ground truth.
    
    curated_seeds = {
        "delivery_delay_tracking": [
            ("Where is my package? Tracking number 9400111899564726274 says delivered yesterday but I checked everywhere.", "AUTO_HANDLE", "Direct customer to check surroundings and provide carrier window before escalation."),
            ("My order has been stuck in 'Departed Facility' in Memphis for 5 days with no new scan.", "AUTO_HANDLE", "Advise carrier transit buffer and offer delivery date check."),
            ("I ordered same day delivery for my medicine and it's 11 PM and nothing has arrived.", "ESCALATE", "Urgent essential item requiring agent investigation with local dispatch."),
            ("Can you tell me which carrier is delivering order #112-9982736-2281920?", "AUTO_HANDLE", "Provide carrier identification and tracking link."),
            ("Tracking shows package handed to resident but I was out of town all week!", "ESCALATE", "Potential misdelivery or theft requiring driver inquiry and claim filing."),
            ("Why does Amazon logistics keep delivering to the back alley instead of front porch?", "AUTO_HANDLE", "Provide instructions on updating delivery preferences and gate codes."),
            ("The delivery driver marked my package undeliverable due to dog, but I don't own any pets.", "AUTO_HANDLE", "Explain re-attempt schedule and update access notes."),
            ("Is there a delay on shipments heading to the Chicago area due to snowstorms?", "AUTO_HANDLE", "Confirm weather-related regional delays and revised delivery window."),
            ("My package was supposed to arrive by 8pm today. It's now 8:15pm, is it still coming?", "AUTO_HANDLE", "Explain delivery routes can run until 9:00 PM."),
            ("Order #701-4491028-1120481 says out for delivery since 6 AM but hasn't arrived.", "AUTO_HANDLE", "Reassure that out for delivery status remains active until end of route."),
            ("Package was left in the rain and now the cardboard box is soaking wet.", "ESCALATE", "Courier handling complaint with potential damaged contents."),
            ("Can I change the delivery address while the package is already in transit?", "AUTO_HANDLE", "Explain courier policy that in-transit address re-routing cannot be modified."),
            ("Tracking says 'Held at carrier facility per customer request' but I never requested this!", "ESCALATE", "Carrier discrepancy requiring agent contact with station."),
            ("Does Amazon Prime deliver packages on Sunday in postal code 98101?", "AUTO_HANDLE", "Confirm Sunday delivery availability for Prime eligible zip codes."),
            ("My tracking status says 'Returned to sender - Damaged in transit'. What happens now?", "AUTO_HANDLE", "Explain that return to sender triggers an automatic refund or re-order upon scan."),
            ("I need my suit delivered before 2 PM tomorrow for a wedding. Can you guarantee delivery?", "ESCALATE", "Time-sensitive guarantee request requiring specialist dispatch review."),
            ("Driver took a delivery photo showing a front door that is definitely not my house.", "ESCALATE", "Confirmed misdelivery to wrong address requiring driver dispatch check."),
            ("Why did my single order get split into four separate shipments with different delivery dates?", "AUTO_HANDLE", "Explain multi-warehouse fulfillment logistics and separate tracking."),
            ("How do I add a gate code so the driver can access my apartment building?", "AUTO_HANDLE", "Provide step-by-step instructions to edit Your Addresses > Delivery Instructions."),
            ("The USPS tracking number provided on Amazon does not exist on the USPS website.", "AUTO_HANDLE", "Advise that tracking numbers take up to 24-48 hours after label creation to register on carrier sites."),
            ("My package has been out for delivery for two consecutive days without arriving.", "ESCALATE", "Repeated missed delivery requiring depot escalation."),
            ("Can I request the driver not ring the doorbell because of sleeping baby?", "AUTO_HANDLE", "Guide user on configuring delivery preferences in their account."),
            ("Order says delivered to Amazon Locker, how long do I have to pick it up?", "AUTO_HANDLE", "State standard 3-calendar-day locker pickup window before automatic return."),
            ("Where do I find the pickup code for my Amazon Hub Locker delivery?", "AUTO_HANDLE", "Explain that the 6-digit pickup code or barcode is sent via email and Amazon app."),
            ("My delivery was marked handed to receptionist, but our office building has been closed for renovation.", "ESCALATE", "False delivery confirmation requiring carrier investigation.")
        ],
        "refund_return_request": [
            ("How do I return a pair of shoes that don't fit? Where do I drop them off?", "AUTO_HANDLE", "Explain 30-day return policy and Kohl's/UPS QR code drop-off procedure."),
            ("I dropped off my return at Whole Foods 3 days ago. When will the refund hit my card?", "AUTO_HANDLE", "State standard 3-5 business day processing timeline after drop-off scan."),
            ("Can I return an opened electronic item if I threw away the original outer box?", "AUTO_HANDLE", "Clarify return policy guidelines regarding packaging and potential restocking terms."),
            ("I want a full refund for an order that was cancelled by the seller.", "AUTO_HANDLE", "Explain that authorization holds on cancelled orders drop within 3-5 business days."),
            ("The return window for my textbook closed yesterday, can an exception be made?", "ESCALATE", "Policy exception request requiring human supervisor discretion."),
            ("I returned two different orders in the same box by accident. How do I get refunds for both?", "ESCALATE", "Warehouse return discrepancy requiring manual tracking reconciliation."),
            ("Can I return a mattress purchased on Amazon? How does return pickup work for heavy items?", "AUTO_HANDLE", "Explain scheduled carrier pickup process for oversized/heavy items."),
            ("I received a gift from someone and want to return it without notifying them for store credit.", "AUTO_HANDLE", "Guide user through the Gift Returns portal using the 17-digit order number."),
            ("Why was a $5.99 return shipping fee deducted from my refund?", "AUTO_HANDLE", "Explain return shipping deduction policy for discretionary customer returns."),
            ("The Amazon returns drop-off QR code won't scan on the kiosk at Kohl's.", "AUTO_HANDLE", "Instruct user to cancel and re-generate return label in Online Returns Center."),
            ("I was promised a refund of $150 by a phone agent last week but received nothing.", "ESCALATE", "Unfulfilled agent commitment requiring CRM interaction history audit."),
            ("Can I get a refund directly to my PayPal account instead of the original credit card?", "AUTO_HANDLE", "State standard policy that refunds must return to the original payment method or gift card balance."),
            ("How long do I have to return an item bought during the holiday shopping season?", "AUTO_HANDLE", "Detail the extended holiday return policy window."),
            ("I accidentally returned my personal sunglasses inside the return box!", "ESCALATE", "Personal property lost at fulfillment center requiring facility contact."),
            ("Is there a limit on how many items I can return in a single month?", "AUTO_HANDLE", "Explain fair return policy guidelines."),
            ("The return tracking shows item delivered to returns center in Kentucky 10 days ago with no refund.", "ESCALATE", "Exceeded standard refund processing SLA requiring manual refund issuance."),
            ("Can I exchange an item for a different size without paying price difference if sale ended?", "ESCALATE", "Price honor request on exchange requiring agent override."),
            ("How do I print a return label if I do not own a printer?", "AUTO_HANDLE", "Explain printerless return options at UPS Store, Kohl's, and Staples."),
            ("I want to cancel an order I placed 10 minutes ago.", "AUTO_HANDLE", "Direct customer to Orders page > Cancel Items button prior to dispatch."),
            ("The seller refuses to issue my return authorization for a third-party marketplace purchase.", "ESCALATE", "Third-party seller dispute requiring Amazon A-to-z Guarantee claim filing."),
            ("Do I have to pay return shipping if the item was not as described?", "AUTO_HANDLE", "Clarify that return shipping is free for defective or incorrectly described products."),
            ("I was charged for an item I returned two weeks ago because Amazon says they never received it.", "ESCALATE", "Chargeback/re-billing dispute requiring proof of drop-off tracking verification."),
            ("Can I drop off a return at any Amazon Fresh store?", "AUTO_HANDLE", "Confirm return drop-off capability at participating Amazon Fresh locations."),
            ("Where can I view the itemized breakdown of my issued refund?", "AUTO_HANDLE", "Guide customer to Your Orders > Order Details > Refund Summary."),
            ("I received a refund confirmation email but the amount is $20 less than what I paid.", "ESCALATE", "Disputed refund deduction requiring billing review.")
        ],
        "damaged_defective_item": [
            ("I opened the parcel and the ceramic teapot was smashed into dozens of pieces.", "AUTO_HANDLE", "Apologize for transit damage and initiate free replacement order."),
            ("The laptop I received powers on but the display has dead vertical colored lines.", "ESCALATE", "High-value defective electronic item requiring tech assessment or replacement."),
            ("I ordered a size Large men's winter coat, but you sent me a toddler's jacket.", "AUTO_HANDLE", "Apologize for fulfillment error and guide re-order/free return process."),
            ("The seal on the vitamins was completely broken and pills were scattered in the envelope.", "AUTO_HANDLE", "Issue immediate safety refund/replacement for contaminated ingestible product."),
            ("One of the four dining chairs in the set arrived with a cracked wooden leg.", "ESCALATE", "Partial furniture damage requiring manufacturer part request or partial credit."),
            ("The cosmetics bottle leaked all over the rest of the items in my delivery box.", "AUTO_HANDLE", "Apologize and issue replacement/refund for spoiled items without requiring return of liquid mess."),
            ("The drill set arrived missing the battery and charger shown in the product picture.", "ESCALATE", "Missing accessory dispute requiring parts fulfillment or full return."),
            ("The phone charger cord sparked and melted when plugged into the wall outlet!", "ESCALATE", "Safety and electrical hazard requiring immediate escalation to product safety team."),
            ("The vinyl record I bought has a deep scratch right across Side A.", "AUTO_HANDLE", "Provide hassle-free replacement label for damaged media."),
            ("The perfume bottle was completely empty because the nozzle broke during shipping.", "AUTO_HANDLE", "Process hazardous material returnless refund per policy for liquids/flammables."),
            ("The packaging box was completely intact, but the smartphone inside had a cracked screen.", "ESCALATE", "High value internal damage requiring warehouse serialization audit."),
            ("The canned food I received has an expiration date from six months ago.", "AUTO_HANDLE", "Issue immediate refund for expired grocery item per freshness guarantee."),
            ("You sent me a UK plug version of the coffee machine instead of the US plug.", "AUTO_HANDLE", "Recognize wrong SKU delivery and provide free return/exchange instructions."),
            ("The blender motor smells like burning plastic after 30 seconds of use.", "AUTO_HANDLE", "Offer manufacturer warranty info or 30-day Amazon defective replacement."),
            ("The zipper on this backpack broke the very first time I pulled it.", "AUTO_HANDLE", "Facilitate return/replacement under 30-day return policy."),
            ("The bicycle arrived with a bent front fork that prevents the wheel from spinning.", "ESCALATE", "Large defective item requiring freight return authorization."),
            ("I received an envelope that was completely unsealed and completely empty inside!", "ESCALATE", "Stolen or lost in transit contents requiring investigation and replacement."),
            ("The book arrived with water damage and wrinkled, illegible pages.", "AUTO_HANDLE", "Initiate replacement copy shipment."),
            ("The digital camera lens has internal fungus and dust despite being sold as 'Brand New'.", "ESCALATE", "Misrepresented condition by third-party seller requiring marketplace compliance review."),
            ("The package smelled like laundry detergent because another item crushed inside leaked on it.", "AUTO_HANDLE", "Process replacement for contaminated order."),
            ("The television screen has a spiderweb crack across the entire 65-inch panel.", "ESCALATE", "High-value freight damaged merchandise requiring specialized carrier inspection."),
            ("The lightbulbs were shattered into glass dust inside the bubble wrap.", "AUTO_HANDLE", "Advise safe disposal and issue replacement without requiring return of broken glass."),
            ("The smartwatch won't charge or turn on straight out of the sealed box.", "AUTO_HANDLE", "Offer standard electronic troubleshooting steps or replacement."),
            ("The hard drive makes clicking noises and is not recognized by any computer.", "AUTO_HANDLE", "Authorize defective return for storage drive."),
            ("You shipped me the Spanish language edition of the textbook instead of English.", "AUTO_HANDLE", "Arrange free return and dispatch correct language edition.")
        ],
        "payment_billing_issue": [
            ("I was charged twice ($64.99 each) for order #114-8829102 on my Visa card.", "ESCALATE", "Duplicate billing dispute requiring merchant ledger verification."),
            ("My credit card was declined at checkout but my bank confirmed my card has no hold.", "AUTO_HANDLE", "Provide instructions to verify billing address match and re-enter card details."),
            ("There is an unauthorized charge of $14.99 from 'Amazon Digital' on my bank statement.", "ESCALATE", "Suspected unauthorized fraudulent transaction requiring account audit."),
            ("My $50 Amazon gift card says 'already redeemed to another account' when I enter the claim code.", "ESCALATE", "Stolen or compromised gift card claim code requiring fraud investigation."),
            ("How do I update the expiration date and CVV for my default payment method?", "AUTO_HANDLE", "Guide user to Manage Payment Methods under Account settings."),
            ("Why is there a temporary $1 authorization hold on my bank account?", "AUTO_HANDLE", "Explain standard pre-authorization verification hold that automatically drops in a few days."),
            ("Can I split a payment between two different credit cards for a single order?", "AUTO_HANDLE", "Explain that Amazon allows split payment between Amazon Gift Card balance and one credit card."),
            ("My promotional discount code 'SAVE20' is not applying at checkout.", "AUTO_HANDLE", "Clarify promotional terms (eligible items, sold by Amazon, expiration)."),
            ("I received an email stating my payment revision was needed or my order will cancel.", "AUTO_HANDLE", "Guide user to click 'Revise Payment' under Your Orders."),
            ("Can I pay for my Amazon order using Apple Pay or Google Pay?", "AUTO_HANDLE", "State supported payment methods and explain lack of direct Apple/Google Pay in web cart."),
            ("I returned an item paid with a gift card. Where did the refund go?", "AUTO_HANDLE", "Explain that gift card payments are refunded back to the Amazon Gift Card balance."),
            ("A seller is asking me to pay them directly via wire transfer to complete the order.", "ESCALATE", "Off-platform payment fraud violation requiring immediate security report."),
            ("Why was sales tax charged on an item shipped to an address in Delaware?", "AUTO_HANDLE", "Provide tax calculation details based on item category and shipping destination."),
            ("My bank blocked the transaction thinking Amazon was fraud. How do I retry without cancelling?", "AUTO_HANDLE", "Advise calling bank to authorize merchant, then clicking Retry Payment in Your Orders."),
            ("How do I download a formal VAT tax invoice for my business accounting?", "AUTO_HANDLE", "Direct user to Your Orders > Printable Order Summary / Invoice link."),
            ("I see recurring charges of $9.99 every month from Amazon that I never signed up for.", "ESCALATE", "Unidentified subscription dispute requiring subscription audit."),
            ("Can I use an international credit card issued in France on Amazon.com?", "AUTO_HANDLE", "Explain international credit card acceptance and Amazon Currency Converter."),
            ("My cashback reward points were deducted but the order did not go through.", "ESCALATE", "Reward points sync failure requiring points reconciliation."),
            ("Can I remove a saved credit card from my account so nobody else can use it?", "AUTO_HANDLE", "Instruct user on navigating to Your Payments > select card > Edit > Remove from wallet."),
            ("Why did the total price in my cart increase right before I clicked Place Order?", "AUTO_HANDLE", "Explain real-time price updates, estimated shipping additions, and dynamic tax."),
            ("My employer gave me a corporate credit card that requires PO numbers. Where do I enter this?", "AUTO_HANDLE", "Explain Amazon Business account features for Purchase Order numbering."),
            ("I was charged an overdraft fee by my bank because Amazon billed me days after I ordered.", "ESCALATE", "Customer grievance regarding authorization timing versus shipment billing."),
            ("Can I pay for groceries on Amazon Fresh using SNAP EBT benefits?", "AUTO_HANDLE", "Explain SNAP EBT eligibility, registration steps, and eligible food products."),
            ("I accidentally selected the wrong credit card for my order. Can I change it after placing?", "AUTO_HANDLE", "Explain that payment method can be changed in Your Orders if order hasn't entered dispatch."),
            ("I have $150 in promotional credit that disappeared from my account overview.", "ESCALATE", "Promotional ledger audit requiring customer service adjustment.")
        ],
        "account_login_access": [
            ("I forgot my Amazon password and the reset email is not arriving in my inbox.", "AUTO_HANDLE", "Advise checking Spam/Junk folders, whitelist Amazon domain, and try SMS reset."),
            ("I changed my phone number and now I cannot receive the Two-Factor OTP to sign in.", "ESCALATE", "Account recovery lockout requiring identity verification document submission."),
            ("I received an email saying someone logged into my account from a device in Russia.", "ESCALATE", "Active security breach requiring immediate account lockdown and password reset."),
            ("My account has been put on hold and customer service asked for billing statement upload.", "ESCALATE", "Account specialist hold requiring secure portal document upload review."),
            ("How do I enable Two-Step Verification using an Authenticator app like Google Authenticator?", "AUTO_HANDLE", "Provide step-by-step guide to Login & Security > Two-Step Verification settings."),
            ("I keep getting an 'Invalid credentials' error even though I know my password is correct.", "AUTO_HANDLE", "Suggest clearing browser cache/cookies or using password reset link."),
            ("Can I merge two separate Amazon accounts registered under different emails into one?", "AUTO_HANDLE", "Explain that accounts cannot be merged, but Family Library sharing can link benefits."),
            ("How do I permanently delete my Amazon account and personal data?", "AUTO_HANDLE", "Guide user to the Close Your Amazon Account privacy request page."),
            ("My 2FA OTP prompt keeps looping back to the password screen without logging me in.", "AUTO_HANDLE", "Suggest incognito mode or disabling conflicting browser extensions."),
            ("An unknown person changed the primary email address on my Amazon account!", "ESCALATE", "Hostile account takeover requiring emergency security escalation."),
            ("How do I change my account name and public profile handle?", "AUTO_HANDLE", "Instruct user to visit Your Account > Your Public Profile > Edit Profile."),
            ("Can I set up Touch ID or Face ID passkey to sign into the Amazon mobile app?", "AUTO_HANDLE", "Explain Passkey configuration in Login & Security settings."),
            ("I am receiving continuous verification codes on my phone that I did not trigger.", "ESCALATE", "Credential stuffing / brute-force alert requiring immediate credential change."),
            ("How do I view all active devices currently logged into my Amazon account?", "AUTO_HANDLE", "Direct user to Manage Your Content and Devices > Devices tab."),
            ("My account was suspended for suspected review manipulation which is completely false.", "ESCALATE", "Policy appeal requiring community trust & safety team review."),
            ("Can I transfer my Kindle book library to a different Amazon account?", "AUTO_HANDLE", "Explain digital rights policy regarding non-transferability between accounts."),
            ("I am locked out because the system asks for answers to security questions I never set.", "ESCALATE", "Auth lock requiring manual identity verification."),
            ("How do I remove an ex-partner from my Amazon Household sharing group?", "AUTO_HANDLE", "Guide user to Manage Your Household > Remove Member settings."),
            ("Why does Amazon ask me to verify my identity every single time I log in from my laptop?", "AUTO_HANDLE", "Explain trusted device cookies and IP location verification safeguards."),
            ("I lost access to the email domain where my Amazon account was registered 10 years ago.", "ESCALATE", "Lost domain credential recovery requiring support verification."),
            ("Can I switch my Amazon account from a personal account to an Amazon Business account?", "AUTO_HANDLE", "Provide link and instructions to register for Amazon Business."),
            ("The CAPTCHA on the login screen is unreadable even after refreshing 10 times.", "AUTO_HANDLE", "Suggest using the audio CAPTCHA option or switching browsers."),
            ("How do I download a copy of all personal data Amazon holds about me under GDPR/CCPA?", "AUTO_HANDLE", "Guide user to Request Your Information page under Privacy Central."),
            ("My child made in-app purchases on the tablet without permission because parental control failed.", "ESCALATE", "Unauthorized minor purchase requiring parental refund review."),
            ("My account was flagged for suspicious activity and my open orders were all cancelled.", "ESCALATE", "Account restriction requiring risk operations verification.")
        ],
        "subscription_prime_issue": [
            ("Why did Amazon charge me $139 for Prime membership when I never authorized renewal?", "AUTO_HANDLE", "Explain auto-renewal policy and offer guide to cancel for full refund if benefits unused."),
            ("How do I cancel my Amazon Prime subscription and get a refund for remaining months?", "AUTO_HANDLE", "Provide instructions to End Membership under Manage Prime Membership."),
            ("I am a university student, how do I apply for the 50% Prime Student discount?", "AUTO_HANDLE", "Direct customer to Prime Student verification page using .edu email."),
            ("I paid for Amazon Prime but Prime Video says I need to purchase a separate subscription.", "AUTO_HANDLE", "Troubleshoot Prime account matching between Amazon retail and Prime Video app."),
            ("Can I share my Amazon Prime delivery benefits with my spouse without paying extra?", "AUTO_HANDLE", "Explain Amazon Household setup to share Prime shipping benefits."),
            ("Why is my Prime delivery taking 4 days instead of 2-day delivery guaranteed by Prime?", "AUTO_HANDLE", "Explain that 2-day transit window applies from the moment item ships from warehouse."),
            ("My Prime subscription auto-renewed yesterday. I haven't used any benefits, can I get a refund?", "AUTO_HANDLE", "Confirm automated full refund policy if zero Prime benefits used since renewal."),
            ("How do I pause my Prime membership while I am traveling abroad for 3 months?", "AUTO_HANDLE", "Explain membership pause functionality in Manage Prime settings."),
            ("Does Prime Video include access to HBO/Max or are those extra channel add-ons?", "AUTO_HANDLE", "Clarify third-party Prime Video Channels require additional subscription fees."),
            ("I was charged $14.99 for Prime after signing up for what was advertised as a free 30-day trial.", "ESCALATE", "Billing discrepancy on free trial promotion requiring agent refund."),
            ("How do I switch my Prime billing from monthly ($14.99) to annual ($139)?", "AUTO_HANDLE", "Guide customer to Payment Plan settings inside Manage Prime."),
            ("Can I pay for my Amazon Prime membership using an Amazon gift card balance?", "AUTO_HANDLE", "Explain policy that Prime memberships require a valid credit/debit card on file."),
            ("My Prime Gaming loot rewards are not linking to my Twitch account.", "AUTO_HANDLE", "Provide link and steps to re-link Amazon and Twitch accounts in Prime Gaming."),
            ("Why am I seeing advertisements on Prime Video when I pay for a Prime membership?", "AUTO_HANDLE", "Explain updated Prime Video tier with ad-free option add-on."),
            ("I cancelled Prime 2 weeks ago but my credit card was just billed again today.", "ESCALATE", "Recurring billing error on cancelled account requiring ledger audit."),
            ("Does Amazon Prime include free Grubhub+ food delivery membership?", "AUTO_HANDLE", "Confirm ongoing Grubhub+ partnership benefit for active Prime members."),
            ("How do I invite a teen to my Amazon Prime Household account?", "AUTO_HANDLE", "Guide customer through Amazon Teen logins under Household management."),
            ("I receive government assistance (EBT/Medicaid), is there a discounted Prime rate?", "AUTO_HANDLE", "Explain Prime Access program offering 50% discount for qualifying recipients."),
            ("Can I download Prime Video movies to watch offline on my laptop?", "AUTO_HANDLE", "Explain offline viewing capabilities via the official Prime Video Windows/Mac app."),
            ("My Prime Music app only lets me shuffle songs now instead of playing on demand.", "AUTO_HANDLE", "Explain Amazon Music Prime catalog shuffle features vs Amazon Music Unlimited tier."),
            ("I was double billed for Prime on two different credit cards this month.", "ESCALATE", "Duplicate subscription charge requiring account deduplication."),
            ("How do I claim the free monthly Twitch channel subscription with Prime?", "AUTO_HANDLE", "Provide steps to click 'Subscribe with Prime' on creator channel."),
            ("Can I gift an Amazon Prime membership to a friend as a birthday present?", "AUTO_HANDLE", "Direct user to Give the Gift of Prime purchase page."),
            ("My company has Amazon Business Prime, does that cover my personal home address?", "AUTO_HANDLE", "Explain separation between Business Prime organization and personal accounts."),
            ("I was promised a 1-month free Prime extension due to late deliveries, where is it?", "ESCALATE", "Discretionary courtesy credit verification requiring support record check.")
        ],
        "product_inquiry_availability": [
            ("Will this OtterBox case for iPhone 14 also fit an iPhone 15?", "AUTO_HANDLE", "Check and explain dimensional/camera cutout differences between phone generations."),
            ("When will the Sony WH-1000XM5 headphones in Silver be restocked?", "AUTO_HANDLE", "Explain stock availability and suggest using 'Email Me When Available' feature."),
            ("Is this product sold directly by Amazon.com or an independent third-party seller?", "AUTO_HANDLE", "Guide customer to inspect 'Ships from' and 'Sold by' on the product buy box."),
            ("What is the exact height and width of this bookshelf assembled?", "AUTO_HANDLE", "Direct customer to the Technical Details / Product Dimensions section on product page."),
            ("Does this wireless gaming mouse come with the USB dongle receiver in the box?", "AUTO_HANDLE", "Confirm included accessories from the manufacturer package contents specification."),
            ("Is this food processor 110V for North America or 220V dual voltage?", "AUTO_HANDLE", "Check electrical voltage specifications in product overview."),
            ("Does the manufacturer warranty apply when buying this watch from a third-party seller?", "AUTO_HANDLE", "Explain authorized dealer versus unauthorized marketplace seller warranty caveats."),
            ("Can this espresso maker use pre-ground coffee or only whole espresso beans?", "AUTO_HANDLE", "Clarify brewing mechanism and filter basket compatibility."),
            ("Are these sunglasses guaranteed authentic Ray-Ban or could they be counterfeit?", "AUTO_HANDLE", "Reassure Amazon anti-counterfeiting policy and verify 'Sold by Amazon' authenticity."),
            ("What is the weight capacity of this steel office chair?", "AUTO_HANDLE", "Provide maximum weight rating from manufacturer specifications."),
            ("Can I pre-order the upcoming video game release and will I get release-day delivery?", "AUTO_HANDLE", "Explain Pre-order Price Guarantee and Release-Day Delivery terms."),
            ("Are the dimensions listed for the package or for the actual product itself?", "AUTO_HANDLE", "Distinguish between Package Dimensions and Item Dimensions in spec table."),
            ("Does this TV mount support VESA 400x400 pattern?", "AUTO_HANDLE", "Check VESA mounting bracket compatibility standards on product page."),
            ("Is this protein powder certified gluten-free and third-party tested?", "AUTO_HANDLE", "Refer to allergen disclaimer and product label certifications."),
            ("Why is the price of this laptop $200 cheaper on Amazon than Best Buy?", "AUTO_HANDLE", "Explain dynamic marketplace pricing and seller promotions."),
            ("Can I use this vacuum cleaner on high-pile shag carpets?", "AUTO_HANDLE", "Consult product user manual recommendations regarding surface compatibility."),
            ("Does this outdoor camera require a monthly cloud subscription to store footage?", "AUTO_HANDLE", "Clarify local SD card storage options versus cloud plan requirements."),
            ("Is there a difference between the US version and the International version of this phone?", "AUTO_HANDLE", "Explain warranty coverage and cellular frequency band differences."),
            ("When will the Black Friday deal pricing on Kindle Paperwhite become active?", "AUTO_HANDLE", "Advise that promotional deals go live according to scheduled event timers."),
            ("Does this jacket run true to size or should I size up for layering?", "AUTO_HANDLE", "Suggest consulting customer sizing widget and review fit feedback."),
            ("Is this replacement refrigerator water filter OEM certified by Samsung?", "AUTO_HANDLE", "Check NSF certification and OEM genuine branding."),
            ("Can I request custom engraving on this jewelry piece ordered through Amazon?", "AUTO_HANDLE", "Explain customization options if 'Customize Now' button is present."),
            ("Will this car battery jump starter work on an 8-cylinder diesel truck engine?", "AUTO_HANDLE", "Check peak amperage and engine liter rating specifications."),
            ("Does the mattress arrive rolled in a box or fully expanded?", "AUTO_HANDLE", "Confirm bed-in-a-box compressed shipping and expansion timeframe."),
            ("Are software license keys emailed instantly upon digital purchase?", "AUTO_HANDLE", "Direct customer to 'Your Games and Software Library' for instant digital keys.")
        ],
        "general_complaint_feedback": [
            ("Your delivery driver literally threw my box from 15 feet away onto my stone porch!", "ESCALATE", "Serious delivery misconduct complaint requiring logistics driver dispatch report."),
            ("I have contacted support 5 times regarding the same issue and received 5 conflicting answers.", "ESCALATE", "Severe customer service failure requiring senior supervisory resolution."),
            ("The delivery van backed up over my lawn and destroyed our sprinkler head!", "ESCALATE", "Property damage claim requiring Amazon Logistics Claims department investigation."),
            ("An agent on chat was extremely disrespectful and closed the chat window in my face.", "ESCALATE", "Agent conduct grievance requiring supervisor review and coaching audit."),
            ("Your company is holding my money hostage for over a month and I am consulting my attorney.", "ESCALATE", "Legal escalation and protracted financial withholding requiring specialist team."),
            ("The driver refused to follow the delivery instructions and left high-value goods on the sidewalk.", "ESCALATE", "Safety and negligence complaint requiring driver feedback citation."),
            ("I was promised a supervisor callback within 2 hours yesterday and never heard back.", "ESCALATE", "Broken commitment requiring immediate supervisory outbound contact."),
            ("Your packaging is ridiculously wasteful—a tiny lipstick arrived in a giant cardboard box.", "AUTO_HANDLE", "Acknowledge environmental feedback with empathy and log packaging feedback."),
            ("I received a delivery at 4:30 AM that woke up my entire house and set off my alarms.", "ESCALATE", "Quiet hours delivery violation requiring delivery time preference enforcement."),
            ("Your automated bot was completely useless and wasted 30 minutes of my time.", "AUTO_HANDLE", "Acknowledge bot limitation with empathy and connect customer to human assistance."),
            ("A driver opened my front door and set the package inside my hallway without permission!", "ESCALATE", "Severe trespassing/safety boundary breach requiring immediate security review."),
            ("I am closing my 15-year-old account because your service has completely degraded.", "ESCALATE", "High-value customer churn risk requiring retention specialist review."),
            ("Why did the delivery driver mark my business address closed at 2 PM on a Tuesday?", "AUTO_HANDLE", "Log delivery discrepancy and coordinate redelivery within business hours."),
            ("The customer service representative could barely understand basic English and hung up.", "ESCALATE", "Support quality complaint requiring review of contact transcript."),
            ("Your driver blocked my driveway for 45 minutes so I missed an important doctor appointment.", "ESCALATE", "Courier misconduct complaint requiring local depot operations manager contact."),
            ("I feel completely insulted by the automated responses I've been receiving to my complaint.", "ESCALATE", "Escalation request requiring personalized human resolution."),
            ("The delivery person was smoking right in front of our children while handing us the package.", "ESCALATE", "Unprofessional conduct complaint requiring driver discipline investigation."),
            ("I want to file a formal complaint with corporate headquarters regarding deceptive billing.", "ESCALATE", "Formal corporate grievance requiring regulatory/executive customer relations."),
            ("Your website has had a checkout error for three straight days and nobody cares.", "AUTO_HANDLE", "Log technical feedback with web engineering team and offer troubleshooting."),
            ("Amazon customer support used to be world class, now it is the worst in the industry.", "AUTO_HANDLE", "Empathize with customer sentiment, thank them for loyalty, and offer specific help."),
            ("The driver left my package right in front of the outward swinging storm door so I was trapped.", "AUTO_HANDLE", "Acknowledge hazard and update driver placement instructions."),
            ("An agent promised me a $50 credit for my trouble, and the next agent called me a liar.", "ESCALATE", "Critical dispute regarding agent integrity and promised compensation."),
            ("Your carrier forged my signature on a high-value delivery requiring direct signature.", "ESCALATE", "Forgery allegation on signature-required delivery requiring courier security audit."),
            ("I've been waiting on hold on the telephone for 90 minutes. This is completely unacceptable.", "ESCALATE", "Severe telephony hold escalation requiring priority callback."),
            ("The driver drove across my neighbor's lawn leaving deep tire ruts in the grass.", "ESCALATE", "Third-party property damage requiring logistics claims handler.")
        ]
    }
    
    golden_rows = []
    gold_convo_ids = set()
    sample_id = 1
    
    for intent, samples in curated_seeds.items():
        assert len(samples) == 25, f"Expected 25 samples for {intent}, found {len(samples)}"
        for msg, decision, reason in samples:
            cid = f"gold_amzn_{sample_id:04d}"
            gold_convo_ids.add(cid)
            golden_rows.append({
                "id": sample_id,
                "customer_message": msg,
                "gold_intent": intent,
                "intent": intent,
                "gold_decision": decision,
                "should_escalate": decision,
                "expected_resolution": reason,
                "conversation_id": cid,
                "source": "curated_evaluation_benchmark"
            })
            sample_id += 1
            
    print(f"Total golden evaluation samples created: {len(golden_rows)}")
    
    # Save golden evaluation set to CSV
    fieldnames = [
        "id",
        "customer_message",
        "gold_intent",
        "intent",
        "gold_decision",
        "should_escalate",
        "expected_resolution",
        "conversation_id",
        "source"
    ]
    with open(GOLDEN_SET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in golden_rows:
            writer.writerow(row)
            
    print(f"Golden evaluation set saved to {GOLDEN_SET_PATH}")
    
    # ENFORCE ZERO LEAKAGE: Filter out any potential golden set collisions from retrieval corpus
    gold_msg_signatures = {r["customer_message"].strip().lower() for r in golden_rows}
    clean_historical = []
    leakage_detected = 0
    
    for rec in historical_records:
        rec_msg = rec.get("customer_message", "").strip().lower()
        rec_cid = rec.get("conversation_id", "")
        if rec_msg in gold_msg_signatures or rec_cid in gold_convo_ids:
            leakage_detected += 1
            continue
        clean_historical.append(rec)
        
    with open(HISTORICAL_RESOLUTION_PATH, "w", encoding="utf-8") as f:
        for rec in clean_historical:
            f.write(json.dumps(rec) + "\n")
            
    print(f"Leakage check complete: Removed {leakage_detected} overlapping records. Final retrieval corpus size: {len(clean_historical)}")

if __name__ == "__main__":
    build_golden_evaluation_set()
