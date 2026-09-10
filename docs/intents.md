# Discovered Intent Taxonomy: AmazonHelp Customer Support

This taxonomy was empirically derived by analyzing historical customer interactions from `AmazonHelp` in the Twitter Customer Support dataset.

## Taxonomy Summary

| Intent Key | Description | Frequency Pattern | Typical Escalation Priority |
| :--- | :--- | :--- | :--- |
| `delivery_delay_tracking` | Inquiries regarding delayed, missing, or in-transit packages | High | Low (Automated tracking lookup) |
| `refund_return_request` | Requests for refunds, return shipping labels, drop-off locations | High | Medium (Account specific) |
| `damaged_defective_item` | Reports of broken, defective, or incorrect items received | Medium | Medium (Requires photo / replacement order) |
| `payment_billing_issue` | Double debits, declined cards, gift card redemption errors | Medium | High (Sensitive financial info) |
| `account_login_access` | 2FA/OTP issues, password reset loops, locked/compromised accounts | Medium | High (Security & auth recovery) |
| `subscription_prime_issue`| Prime subscription renewals, unauthorized fees, benefits access | Low-Medium | Medium (Subscription policy) |
| `product_inquiry_availability`| Specifications, compatibility, restock dates, seller info | Low-Medium | Low (Catalog knowledge) |
| `general_complaint_feedback`| Severe courier misconduct, agent dissatisfaction, formal disputes | Medium | High (Immediate escalation required) |

---

## Detailed Intent Specifications

### 1. `delivery_delay_tracking`
- **Description**: Customer seeking whereabouts of an order, tracking number details, or explanation for shipping delay.
- **Inclusion Criteria**: Mentions of "where is my package", "tracking number", "carrier delay", "order not arrived", "USPS/UPS/Amazon Logistics".
- **Exclusion Criteria**: Package arrived but was shattered/defective (`damaged_defective_item`); customer demanding immediate refund due to delay (`refund_return_request`).
- **Historical Support Behavior**: Support asks for order details via DM or provides public tracking assistance, advises checking around property, and sets a 24-48 hour buffer for courier updates.

### 2. `refund_return_request`
- **Description**: Customer requesting return labels, wanting to return an item within the policy window, or inquiring about refund status.
- **Inclusion Criteria**: Mentions of "return label", "how to return", "when will my refund show up", "cancel order for refund", "Kohl's/UPS dropoff".
- **Exclusion Criteria**: Charge disputes on credit cards without returning items (`payment_billing_issue`); Prime subscription refund (`subscription_prime_issue`).
- **Historical Support Behavior**: Support outlines the 30-day return policy, points customer to the Online Returns Center, and states typical banking refund timelines (3–5 business days).

### 3. `damaged_defective_item`
- **Description**: Customer received an item that is broken, defective, spoiled, missing parts, or completely wrong.
- **Inclusion Criteria**: "Arrived broken", "damaged box", "defective unit", "wrong item sent", "glass shattered".
- **Exclusion Criteria**: Package merely delivered late with contents intact (`delivery_delay_tracking`).
- **Historical Support Behavior**: Support expresses empathy, apologizes for condition, initiates replacement shipment or issues returnless refund if applicable.

### 4. `payment_billing_issue`
- **Description**: Unrecognized charges, double billing, payment declines, or gift card balance errors.
- **Inclusion Criteria**: "Charged twice", "unauthorized charge", "payment method rejected", "gift card code invalid".
- **Exclusion Criteria**: Normal refund processing delay (`refund_return_request`).
- **Historical Support Behavior**: Support asks customer to verify pending bank authorizations, instructs them to check Your Payments in account settings, and directs sensitive card details to secure DM channels.

### 5. `account_login_access`
- **Description**: Inability to log into Amazon account due to OTP/2FA failure, password resets, or account lockouts.
- **Inclusion Criteria**: "Cannot log in", "no OTP code received", "2FA loop", "account suspended", "password reset link broken".
- **Exclusion Criteria**: Inquiries that occur while logged into account without access issues.
- **Historical Support Behavior**: Directs user to the Two-Step Verification account recovery workflow, requests secure customer service contact via verified email, escalates to account specialists.

### 6. `subscription_prime_issue`
- **Description**: Questions or disputes concerning Amazon Prime membership fees, renewal notices, student plans, or streaming benefits.
- **Inclusion Criteria**: "Prime fee charged", "cancel my Prime", "student discount", "Prime video not working".
- **Exclusion Criteria**: Order queries that just happen to use Prime delivery (`delivery_delay_tracking`).
- **Historical Support Behavior**: Support explains Prime benefit terms, directs customer to Manage Prime Membership page to trigger pro-rated cancellations, or troubleshoots Prime Video sign-in.

### 7. `product_inquiry_availability`
- **Description**: Pre-purchase questions regarding product compatibility, specifications, dimensions, stock levels, or seller authenticity.
- **Inclusion Criteria**: "Is this compatible with", "when will this be back in stock", "sold by Amazon or 3rd party", "what are dimensions".
- **Exclusion Criteria**: Customer has already received the item and found it broken (`damaged_defective_item`).
- **Historical Support Behavior**: Refers to Product Details table, suggests adding out-of-stock items to Wishlist for alerts, clarifies Fulfilled by Amazon terms.

### 8. `general_complaint_feedback`
- **Description**: Expressing strong dissatisfaction with service, rude delivery driver behavior, poor customer support, or filing formal complaints.
- **Inclusion Criteria**: "Horrible service", "driver threw my box", "rude agent", "filing a complaint", "cancelling my account out of disgust".
- **Exclusion Criteria**: Routine inquiries expressed with mild politeness.
- **Historical Support Behavior**: Immediate empathy and de-escalation; promises to log feedback with logistics leadership; prompt human escalation for investigation.
