#!/usr/bin/env python3
"""
generate_docs.py — Generate Everstorm Outfitters PDF knowledge base.

Produces 15 PDFs in data/:
  - 12 core policy documents (same corpus as vector_rag_chatbot)
  - 3 cross-document PDFs added for GraphRAG — these explicitly reference
    entities from other documents, creating the inter-document relationships
    that graph traversal can exploit.

Usage:
    pip install anthropic reportlab python-dotenv
    python generate_docs.py
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from anthropic import Anthropic
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

DATA_DIR = Path(__file__).parent / "data"

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

SYSTEM_PROMPT = """\
You are writing customer support policy documents for Everstorm Outfitters,
a fictional outdoor adventure gear e-commerce company based in Bozeman, MT.

Formatting rules (follow exactly):
- Do NOT include a document header — that is added separately
- Numbered sections use format: "1  Section Title" (digit, two spaces, title in Title Case)
- Bullet points use "  • text" (two spaces, bullet, space, text)
- Tables use pipe format with a separator row:
    | Col1 | Col2 | Col3 |
    | --- | --- | --- |
    | val | val | val |
- No markdown (no ##, no **bold**, no _italic_)
- Be specific: use real numbers, prices, timeframes, URLs, phone numbers
- All URLs use everstorm.example domain
- All phone numbers use +1 (406) 555-XXXX format
- End every document with a numbered Contact section
- Write 1.5 to 2 pages of content total\
"""

# ── Document definitions ──────────────────────────────────────────────────────
# Documents marked [CROSS-DOC] explicitly reference entities from other
# documents — these are the foundation of GraphRAG multi-hop queries.

DOCUMENTS = [

    # ── Core policy documents ─────────────────────────────────────────────────

    {
        "filename": "Everstorm_Loyalty_Program.pdf",
        "code": "LYL-2025-01",
        "title": "SUMMIT REWARDS LOYALTY PROGRAM",
        "prompt": """\
Write the full body content for an Everstorm Outfitters policy document: Summit Rewards loyalty program.

Sections to include:
1  Program Overview — brief intro to Summit Rewards
2  Enrollment — sign up at everstorm.example/rewards; existing customers auto-enrolled
3  Earning Points — purchases ($1 = 1 pt), product reviews (50 pts), referrals (200 pts per friend who places an order), birthday bonus (100 pts annually)
4  Membership Tiers — table with columns Tier | Points Range | Point Bonus | Perks
   Base Camp | 0–499 | 5% | Standard benefits
   Summit | 500–1,999 | 10% | Free standard shipping on all orders
   Everest | 2,000+ | 15% | Free expedited shipping + early sale access
5  Redeeming Points — 500 pt minimum = $5 reward coupon, no cash value, excluded from Final Sale items, max 1 coupon per order
6  Points Expiration — points expire after 18 months of account inactivity; reactivate with any purchase
7  Checking Your Balance — everstorm.example/account dashboard
8  Contact — rewards@everstorm.example  •  +1 (406) 555-0201\
""",
    },
    {
        "filename": "Everstorm_Gift_Cards.pdf",
        "code": "GFC-2025-02",
        "title": "GIFT CARDS — PURCHASE, REDEMPTION & BALANCE",
        "prompt": """\
Write the full body content for an Everstorm Outfitters policy document: gift cards.

Sections to include:
1  Available Denominations — $25, $50, $100, $150, $200, custom up to $500
2  How to Purchase — online at everstorm.example/gift-cards or at any flagship store
3  Delivery — digital: email within 15 minutes; physical: mailed in 3–5 business days, gift packaging option ($3)
4  How to Redeem — enter code at checkout; partial use allowed; combine up to 3 cards per order; cannot use a gift card to purchase another gift card
5  Checking Your Balance — everstorm.example/gift-card-balance (enter card number + PIN)
6  Expiration & Fees — no expiration on balance; physical card must be activated within 12 months of purchase; no dormancy fees
7  Lost or Stolen Cards — contact support with original order number; replacement issued for remaining balance after verification
8  Contact — giftcards@everstorm.example  •  +1 (406) 555-0202\
""",
    },
    {
        "filename": "Everstorm_Order_Cancellation_Policy.pdf",
        "code": "CAN-2025-03",
        "title": "ORDER CANCELLATION POLICY",
        "prompt": """\
Write the full body content for an Everstorm Outfitters policy document: order cancellations.

Sections to include:
1  Cancellation Window — orders cancellable within 30 minutes via account portal
2  How to Cancel — log in at everstorm.example/orders, select order, click Cancel; after 30 min email orders@everstorm.example immediately
3  Cancellation Scenarios — table with columns: Scenario | Time Since Order | Can Cancel? | How
   Within window | < 30 min | Yes | Self-serve portal
   After window, pre-manifest | 30 min–2 hrs | Possible | Email logistics immediately
   After manifest | > 2 hrs | No | Use return process after delivery
4  Pre-Order Cancellations — cancel up to 48 hours before listed ship date
5  Backorder Cancellations — auto-cancelled with full refund if stock not restored in 21 days, unless customer opts to wait via email
6  Refund Timeline — 2–3 business days to original payment method once cancellation confirmed
7  Non-Cancellable Orders — custom embroidered items; digital gift cards already delivered to recipient email
8  Contact — orders@everstorm.example  •  +1 (406) 555-0203\
""",
    },
    {
        "filename": "Everstorm_International_Shipping_Guide.pdf",
        "code": "INT-2025-04",
        "title": "INTERNATIONAL SHIPPING & CUSTOMS GUIDE",
        "prompt": """\
Write the full body content for an Everstorm Outfitters policy document: international shipping and customs.

Sections to include:
1  Regions We Ship To — Americas (US, Canada, Mexico), Europe (EU/EEA, UK, Switzerland), Asia-Pacific (Japan, Singapore, Australia, New Zealand, South Korea)
2  Duties & Taxes — DDU vs DDP explanation; EU ships DDP (duties included); all others ship DDU (customer pays on delivery)
3  Customs Clearance Times — table: Region | Standard Clearance | Expedited Clearance
   EU | 2–4 business days | 1–2 business days
   UK | 1–3 business days | 1 business day
   Australia/NZ | 3–5 business days | 2–3 business days
   Japan/Singapore | 1–2 business days | Next business day
   Canada | 1–3 business days | 1 business day
4  Restricted & Prohibited Items — table: Item | Restricted To
   Aerosol cans | Prohibited to AU/NZ
   Bladed tools (knives, multi-tools) | Restricted in UK (blade > 3 in)
   Lithium battery devices | Require special declaration to EU
   No restrictions | Canada, Japan, Singapore
5  Address Requirements — must match government ID; include apartment/suite; no PO boxes internationally
6  VAT Invoices — available for EU customers within 30 days; request at billing@everstorm.example
7  International Returns — customer pays return shipping (~$15–25 deducted from refund); duties non-refundable; start at everstorm.example/returns
8  Contact — international@everstorm.example  •  +1 (406) 555-0204\
""",
    },
    {
        "filename": "Everstorm_Extended_Warranty.pdf",
        "code": "WAR-2025-05",
        "title": "SUMMIT PROTECT — EXTENDED WARRANTY PROGRAM",
        "prompt": """\
Write the full body content for an Everstorm Outfitters policy document: Summit Protect extended warranty.

Sections to include:
1  What is Summit Protect — extends standard 12-month warranty to 3 years; adds accidental damage coverage
2  Program Cost — table: Product Category | Summit Protect Price
   Footwear | $12
   Apparel (jackets, pants, base layers) | $8
   Technical Gear (packs, tents, sleeping bags) | $20
   Electronics (headlamps, GPS devices) | $25
3  What Is Covered — manufacturing defects, accidental tears/seam failure, zipper failure, sole delamination, buckle/clasp breakage, strap detachment
4  What Is NOT Covered — intentional damage, normal wear and fading, lost items, cosmetic scratches without functional impact, damage from improper care (against care label instructions)
5  How to File a Claim — visit everstorm.example/warranty-claim; submit 3 photos showing defect; 5–7 business day assessment; contacted by email with outcome
6  Claim Resolution — repair first, then same/equivalent replacement, then store credit at original purchase price
7  How to Purchase — add at checkout or within 30 days of purchase via account portal at everstorm.example/account
8  Transferability — transfers with item if gifted (update at warranty@everstorm.example); does not transfer if item is sold
9  Contact — warranty@everstorm.example  •  +1 (406) 555-0205\
""",
    },
    {
        "filename": "Everstorm_Privacy_and_Data_Policy.pdf",
        "code": "PRV-2025-06",
        "title": "PRIVACY & DATA POLICY",
        "prompt": """\
Write the full body content for an Everstorm Outfitters privacy policy written in plain, accessible English (not legalese).

Sections to include:
1  What We Collect — name, email, shipping/billing address, payment method (last 4 digits only), purchase history, browsing behavior, device type and IP address
2  How We Use Your Data — fulfil orders, send shipping updates, personalize recommendations, send marketing emails only if opted in, improve site performance
3  Who We Share It With — payment processors (Stripe, PayPal, Apple Pay), shipping carriers (UPS, DHL, FedEx), analytics (anonymized only); we do NOT sell personal data
4  GDPR Rights (EU Customers) — access, rectify, erase, port, and object to processing; submit requests at everstorm.example/privacy-request (30-day response)
5  CCPA Rights (California Residents) — right to know, delete, and opt-out of sale (we do not sell data); same privacy request portal
6  Cookie Policy — strictly necessary (always on), analytics (opt-out via cookie banner), marketing (opt-in only)
7  Data Retention — account and order data: 7 years for tax/legal compliance; browsing/analytics data: purged after 13 months
8  Deleting Your Account — request at everstorm.example/account/close; 30-day cooldown; order history retained per legal requirements
9  Contact — privacy@everstorm.example  •  DPO: dpo@everstorm.example\
""",
    },
    {
        "filename": "Everstorm_Store_Locations_and_Hours.pdf",
        "code": "STR-2025-07",
        "title": "STORE LOCATIONS, HOURS & IN-STORE SERVICES",
        "prompt": """\
Write the full body content for an Everstorm Outfitters document: physical store locations and services.

Sections to include:
1  Flagship Stores — list all 3 with address, hours, and phone:
   Denver CO: 1420 Larimer St, Denver CO 80202 | Mon–Sat 9am–8pm, Sun 10am–6pm | +1 (303) 555-0301
   Seattle WA: 2211 Pine St, Seattle WA 98121 | Mon–Sat 9am–8pm, Sun 11am–6pm | +1 (206) 555-0302
   Portland OR: 888 NW 23rd Ave, Portland OR 97210 | Mon–Sat 10am–7pm, Sun 11am–5pm | +1 (503) 555-0303
2  Outlet Stores — 2 locations:
   Bozeman MT: 405 W Main St, Bozeman MT 59715 | Mon–Sat 10am–6pm, Sun 11am–5pm
   Salt Lake City UT: 220 S 300 W, Salt Lake City UT 84101 | Mon–Sat 10am–7pm, Sun 12pm–5pm
3  In-Store Services — table: Service | Details | Price
   Gear Fitting | Expert sizing for footwear and packs | Free
   Boot Stretching | Width and length adjustment | $10
   Hemming & Alterations | Pants hem, zipper replacement | $15–$35
   Gear Rental | Day-rate for select items | See rate card in store
4  Buy Online Pick Up In Store (BOPIS) — order online, ready within 2 hours at any flagship store, hold for 5 days, bring order confirmation and photo ID
5  In-Store Returns — same 30-day policy as online; immediate refund to card; no shipping label needed
6  Store Events — monthly gear clinics, athlete meet-and-greets, seasonal trail talks; sign up at everstorm.example/events
7  Accessibility — all locations ADA compliant; accessible parking; staff assist available on request
8  Contact — stores@everstorm.example  •  Store finder: everstorm.example/stores\
""",
    },
    {
        "filename": "Everstorm_Promo_and_Discount_Policy.pdf",
        "code": "PRM-2025-08",
        "title": "PROMOTIONAL DISCOUNTS & COUPON POLICY",
        "prompt": """\
Write the full body content for an Everstorm Outfitters policy document: promotional discounts and coupons.

Sections to include:
1  Types of Discounts — promo codes (email/SMS), loyalty reward coupons (Summit Rewards), student discount (15% via ID.me at everstorm.example/student), military & first responder discount (15% via ID.me), new customer welcome (10% off first order with email signup)
2  How to Apply — enter code in Promo Code field at checkout; one code per order; not case-sensitive
3  Stacking Rules — table: Discount A | Discount B | Can Stack?
   Promo Code | Loyalty Coupon | No
   Promo Code | Summit Rewards Points | Yes
   Student/Military | Sale Price | No
   Loyalty Coupon | Summit Rewards Points | Yes
4  Sale Events — End of Season Sale (January and July), Black Friday/Cyber Monday (last week of November), Flash Sales (48-hour max, email/SMS only), In-Store Gear Swap events
5  Price Adjustment Policy — if item drops in price within 14 days of purchase, contact promo@everstorm.example for a one-time adjustment to everstorm.example price only (no third-party matching)
6  Expired Codes — expire on stated date; no extensions or reactivations
7  Code Abuse Policy — accounts with unusual redemption patterns may be reviewed and lose discount eligibility
8  Contact — promo@everstorm.example  •  +1 (406) 555-0208\
""",
    },
    {
        "filename": "Everstorm_Account_and_Security.pdf",
        "code": "ACC-2025-09",
        "title": "ACCOUNT MANAGEMENT & SECURITY",
        "prompt": """\
Write the full body content for an Everstorm Outfitters policy document: account management and security.

Sections to include:
1  Creating an Account — sign up at everstorm.example/signup; email verification required within 24 hours; or use Google/Apple sign-in
2  Password Requirements — minimum 12 characters; must include 1 uppercase letter, 1 number, 1 special character (!@#$%^&*); may not contain part of your email address
3  Two-Factor Authentication (2FA) — optional but strongly recommended; supported via authenticator app (Google Authenticator, Authy) or SMS; enable at everstorm.example/account/security
4  Forgot Password — reset link emailed within 2 minutes; valid for 60 minutes; single-use
5  Account Lockout — 5 failed attempts triggers 30-minute lockout; then retry or use forgot password flow
6  Security Alerts — we email you for: first login from new device, billing address change, password change, order over $200; reply immediately if unrecognized
7  Stored Payment Methods — payment details tokenized; we store only last 4 digits and card type; full card numbers never stored (PCI DSS Level 1 compliant)
8  Closing Your Account — submit at everstorm.example/account/close; 30-day cooldown period; order history retained 7 years per tax law
9  Social Login — Google and Apple sign-in supported; link/unlink in account settings; add a direct password at any time
10  Contact — security@everstorm.example  •  Fraud hotline: +1 (406) 555-0209 (24/7)\
""",
    },
    {
        "filename": "Everstorm_Sustainability_and_Recycling.pdf",
        "code": "ECO-2025-10",
        "title": "SUSTAINABILITY PROGRAM & GEAR RECYCLING",
        "prompt": """\
Write the full body content for an Everstorm Outfitters document: sustainability and gear recycling program.

Sections to include:
1  Materials Commitment — minimum 85% recycled polyester in all fleece; GOTS-certified organic cotton in base layers; bluesign-approved technical fabrics; PFC-free DWR on all weather shells since 2024
2  Packaging — 100% recycled and recyclable mailers; soy-based inks; no single-use poly bags; packaging weight reduced 40% since 2020
3  Gear Recycle Program — mail back any Everstorm-brand item (any condition) for $10 store credit; items cleaned and donated to nonprofits or broken into recycled fiber
4  How to Participate — request prepaid recycle label at everstorm.example/recycle; limit 5 items per label; 1 label per household per month; credit applied within 5 business days of receipt
5  Carbon Footprint — we offset 110% of all shipping emissions via Gold Standard-verified carbon credits; net-zero manufacturing target by 2030
6  Repair Before Replace — all flagship stores offer repair service; spare buckles, cord-locks, and zipper pulls available free at stores or by mail at everstorm.example/spareparts
7  Annual Impact Report — published each April at everstorm.example/impact; 2024 highlights: 1.2M kg recycled fiber used, 48,000 items diverted from landfill, $380K donated to trail maintenance organizations
8  Certifications — B Corp certified since 2022 (score: 94.2); 1% For The Planet member; Fair Trade Certified manufacturing partners
9  Contact — sustainability@everstorm.example\
""",
    },
    {
        "filename": "Everstorm_B2B_Corporate_Orders.pdf",
        "code": "B2B-2025-11",
        "title": "B2B & CORPORATE ORDERS",
        "prompt": """\
Write the full body content for an Everstorm Outfitters policy document: B2B and corporate orders.

Sections to include:
1  Who Can Apply — businesses, nonprofits, guide services, schools and universities, government agencies, outdoor recreation programs
2  Minimum Order Quantities — table: Category | Min Units Per Style/Color
   Apparel (tops, jackets, pants) | 12 units
   Footwear | 6 pairs
   Accessories (hats, socks, gloves) | 24 units
   Hard Goods (packs, tents) | 6 units
3  Volume Discounts — table: Order Size | Discount
   12–24 units | 10%
   25–49 units | 15%
   50–99 units | 20%
   100+ units | 25% + dedicated account rep
4  Custom Embroidery & Logo — setup fee $45 per logo location; embroidery cost: under 8,000 stitches $8/item, 8,000–15,000 stitches $12/item, over 15,000 stitches $15/item; minimum 12 units per embroidered design
5  Lead Times — in-stock items: 5–7 business days; custom embroidery: 10–14 business days; international B2B: add 5–10 days
6  Payment Terms — credit card or ACH for all orders; NET 30 available for accounts with >$5,000 annual spend (apply at b2b.everstorm.example/credit)
7  Invoicing & PO Numbers — PDF invoices emailed on shipment; PO numbers accepted and printed on invoice; tax-exempt certificate upload at b2b.everstorm.example/tax-exempt
8  Returns — 15-day window; unworn/unused only; custom embroidered items non-returnable unless defective; damaged items must be reported within 48 hours with photos
9  B2B Portal — b2b.everstorm.example: order history, invoices, saved addresses, one-click reorder
10  Contact — b2b@everstorm.example  •  +1 (406) 555-0211 (Mon–Fri 08:00–17:00 MT)\
""",
    },
    {
        "filename": "Everstorm_Accessibility_Services.pdf",
        "code": "ACS-2025-12",
        "title": "ACCESSIBILITY SERVICES & ADAPTIVE GEAR",
        "prompt": """\
Write the full body content for an Everstorm Outfitters document: accessibility services and adaptive gear.

Sections to include:
1  Website Accessibility — WCAG 2.1 AA compliant; screen reader optimized (NVDA, JAWS, VoiceOver tested); full keyboard navigation; high-contrast mode toggle in footer; text resize support; alt text on all product images
2  Adaptive Gear Line — specific products designed for ease of use:
   Summit Shell Jacket — magnetic front closure, eliminates zipper struggle
   TrailReady Pants — elastic waistband with one-hand-friendly drawcord
   Base Layer Set — loop pulls on all zippers, tagless construction
   Summit Gloves — Velcro cuff, no fiddly buckle
   Trekking Pack 32L — magnetic sternum strap buckle, padded easy-grip hip belt
   Approach Shoe — BOA Fit System lacing, no-tie alternative
3  In-Store Assistance — staff trained in adaptive fitting; private fitting rooms at all locations; mobility aid access (ramps, wide aisles, accessible restrooms) at all stores
4  Phone & Chat Ordering — full catalog available by phone +1 (406) 555-0100 (Mon–Fri 8am–7pm MT); staff will process while you stay on the line
5  Large-Print Catalog — available by mail or email PDF on request at access@everstorm.example
6  Accessibility Feedback — report barriers at access@everstorm.example; 7-business-day response commitment; all reports reviewed by product and web teams
7  Community Partnerships — 2% of adaptive gear revenue donated to Challenged Athletes Foundation and Paradox Sports; Everstorm sponsors 10 adaptive athletes annually
8  Contact — access@everstorm.example  •  +1 (406) 555-0212\
""",
    },

    # ── Cross-document PDFs (GraphRAG) ────────────────────────────────────────
    # These documents explicitly reference entities defined in the core policy
    # documents above, creating the inter-document relationships that power
    # multi-hop graph traversal queries.

    {
        "filename": "Everstorm_Member_Benefits_Guide.pdf",
        "code": "MBG-2025-13",
        "title": "SUMMIT REWARDS — COMPLETE MEMBER BENEFITS GUIDE",
        "prompt": """\
Write the full body content for an Everstorm Outfitters document: complete member benefits guide for Summit Rewards loyalty tiers.

This document should explicitly connect the three loyalty tiers (Base Camp, Summit, Everest) to specific benefits across returns, shipping, warranty, and promotions.

Sections to include:
1  Overview — Summit Rewards has three tiers: Base Camp (0–499 pts), Summit (500–1,999 pts), Everest (2,000+ pts); this guide summarizes every benefit by tier
2  Shipping Benefits — table: Benefit | Base Camp | Summit | Everest
   Standard Shipping | $6.99 flat rate | Free on all orders | Free on all orders
   Expedited Shipping (2-day) | $14.99 | $14.99 | Free on all orders
   Overnight Shipping | $24.99 | $24.99 | 50% off ($12.49)
3  Return Benefits — table: Benefit | Base Camp | Summit | Everest
   Return Window | 30 days | 30 days | 45 days extended window
   Return Shipping Label | $6.99 deducted | Free prepaid label | Free prepaid label + priority processing
   Exchanges | Standard process | Standard process | Instant exchange (new item ships before return received)
4  Warranty Benefits — table: Benefit | Base Camp | Summit | Everest
   Standard Warranty | 12 months | 12 months | 12 months
   Summit Protect discount | Full price | 10% off Summit Protect | 20% off Summit Protect + free coverage on items over $150
   Warranty Claim Priority | Standard 5–7 days | Standard 5–7 days | Priority 2–3 day assessment
5  Promotional Benefits — table: Benefit | Base Camp | Summit | Everest
   Point Bonus Rate | 5% bonus | 10% bonus | 15% bonus
   Early Sale Access | No | No | 24-hour early access to all sale events
   Birthday Bonus | 100 pts | 150 pts | 200 pts + surprise gift
   Flash Sale Invitations | No | Yes | Yes + exclusive Everest-only flash sales
6  Points Earning Bonuses by Tier — Base Camp: $1 = 1 pt; Summit: $1 = 1.1 pts; Everest: $1 = 1.15 pts; all tiers earn 50 pts for reviews and 200 pts for referrals
7  How to Check Your Tier — everstorm.example/account/rewards shows current tier, points balance, and progress to next tier
8  Contact — rewards@everstorm.example  •  +1 (406) 555-0201\
""",
    },
    {
        "filename": "Everstorm_Product_Categories_and_Policies.pdf",
        "code": "PCP-2025-14",
        "title": "PRODUCT CATEGORIES, POLICIES & COVERAGE GUIDE",
        "prompt": """\
Write the full body content for an Everstorm Outfitters document: product categories and their applicable policies.

This document explicitly maps each product category to the relevant return policy, warranty coverage, Summit Protect pricing, recycling eligibility, and B2B availability.

Sections to include:
1  Product Category Overview — Everstorm carries five main categories: Footwear, Apparel, Technical Gear, Electronics, and Accessories
2  Footwear — table: Policy | Details
   Return Window | 30 days unworn; 45 days for Everest tier members
   Return Condition | Unworn, original box required for full refund
   Standard Warranty | 12 months; covers sole delamination and stitching failure
   Summit Protect | $12 — extends to 3 years, adds accidental damage
   Gear Recycle Program | Eligible — $10 store credit per pair returned
   B2B Minimum | 6 pairs per style/color
3  Apparel (jackets, pants, base layers, fleece) — table: Policy | Details
   Return Window | 30 days unwashed; 45 days for Everest tier members
   Return Condition | Tags attached, unwashed
   Standard Warranty | 12 months; covers zipper failure and seam separation
   Summit Protect | $8 — extends to 3 years, adds accidental tear coverage
   Gear Recycle Program | Eligible — $10 store credit per item
   B2B Minimum | 12 units per style/color
4  Technical Gear (packs, tents, sleeping bags, climbing gear) — table: Policy | Details
   Return Window | 30 days unused; 45 days for Everest tier members
   Return Condition | Unused, all components included
   Standard Warranty | 12 months; covers buckle, clasp, and zipper failure
   Summit Protect | $20 — extends to 3 years, covers strap detachment and frame damage
   Gear Recycle Program | Eligible — $10 store credit per item
   B2B Minimum | 6 units per style
5  Electronics (headlamps, GPS devices, solar chargers) — table: Policy | Details
   Return Window | 30 days; must include all accessories and original packaging
   Return Condition | Factory reset required; all accessories included
   Standard Warranty | 12 months; covers manufacturing defects only
   Summit Protect | $25 — extends to 3 years, adds accidental damage and water damage
   Gear Recycle Program | Eligible — $10 store credit; batteries removed before shipping
   B2B Minimum | Not available for B2B bulk orders
6  Accessories (hats, socks, gloves, gaiters) — table: Policy | Details
   Return Window | 30 days; unworn, tags attached
   Standard Warranty | 90 days; manufacturing defects only
   Summit Protect | Not available for accessories
   Gear Recycle Program | Eligible — bundle 3+ items for $10 store credit
   B2B Minimum | 24 units per style/color
7  Final Sale Items — any item marked Final Sale is non-returnable, non-exchangeable; Summit Protect still purchasable within 30 days; Gear Recycle Program still applies
8  Contact — support@everstorm.example  •  +1 (406) 555-0100\
""",
    },
    {
        "filename": "Everstorm_Partner_and_Referral_Programs.pdf",
        "code": "PRP-2025-15",
        "title": "PARTNER & REFERRAL PROGRAMS",
        "prompt": """\
Write the full body content for an Everstorm Outfitters document: partner and referral programs.

This document explicitly references Summit Rewards loyalty tiers, the Gear Recycle Program, the B2B corporate program, and student/military discounts — connecting multiple programs into a unified overview.

Sections to include:
1  Program Overview — Everstorm runs four partner programs: Customer Referral (via Summit Rewards), Athlete Ambassador, Nonprofit & Guide Service Partner, and Educational Institution Partner
2  Customer Referral Program — part of Summit Rewards; refer a friend who completes their first order: referrer earns 200 pts; new customer earns 10% off first order; Everest tier members earn 300 pts per referral; no limit on referrals; track at everstorm.example/account/referrals
3  Athlete Ambassador Program — apply at everstorm.example/ambassador; requirements: active outdoor athlete, 1,000+ social followers or local club membership; benefits: 40% gear discount, $500 annual gear credit, early product access, co-branded content opportunities; ambassadors automatically enrolled at Everest tier in Summit Rewards
4  Nonprofit & Guide Service Partner Program — qualifies for B2B corporate pricing (10–25% volume discount); additionally: free Summit Protect on all Technical Gear orders over $500; eligible for Gear Recycle Program with 2x store credit ($20 per item vs standard $10); apply at b2b.everstorm.example/nonprofit
5  Educational Institution Partner Program — schools, universities, outdoor education programs; qualifies for B2B pricing; students in accredited programs receive same 15% student discount as ID.me verified students; instructors receive Everest tier Summit Rewards status while partnership active; annual gear grant ($1,000 gear credit) for qualifying programs
6  Sustainability Partner Network — organizations running trail maintenance or conservation programs; co-branded sustainability initiatives; 5% of partner order revenue donated to Challenged Athletes Foundation or trail organizations of partner's choice; connect at sustainability@everstorm.example
7  How Programs Interact — table: Program | Summit Rewards Tier | B2B Discount | Summit Protect | Gear Recycle Bonus
   Standard Customer | Earned normally | No | Standard price | $10/item
   Everest Tier Member | Everest (2,000+ pts) | No | 20% off | $10/item
   Nonprofit Partner | Everest tier granted | Yes (10–25%) | Free on Tech Gear >$500 | $20/item
   Athlete Ambassador | Everest tier granted | 40% discount | Free on all gear | $20/item
   Educational Partner | Everest tier (instructors) | Yes (B2B rates) | Standard price | $10/item
8  Contact — partners@everstorm.example  •  +1 (406) 555-0215  •  b2b.everstorm.example/partners\
""",
    },
]


# ── Styles ────────────────────────────────────────────────────────────────────

def _styles() -> dict:
    return {
        "company": ParagraphStyle("company", fontName="Helvetica", fontSize=11, spaceAfter=1),
        "title":   ParagraphStyle("title",   fontName="Helvetica-Bold", fontSize=13, spaceAfter=1),
        "code":    ParagraphStyle("code",    fontName="Helvetica", fontSize=9, spaceAfter=14),
        "section": ParagraphStyle("section", fontName="Helvetica-Bold", fontSize=11,
                                  spaceBefore=10, spaceAfter=4),
        "body":    ParagraphStyle("body",    fontName="Helvetica", fontSize=9.5,
                                  leading=14, spaceAfter=3),
        "bullet":  ParagraphStyle("bullet",  fontName="Helvetica", fontSize=9.5,
                                  leading=13, leftIndent=14, spaceAfter=2),
        "tcell":   ParagraphStyle("tcell",   fontName="Helvetica", fontSize=9, leading=12),
        "thdr":    ParagraphStyle("thdr",    fontName="Helvetica-Bold", fontSize=9, leading=12),
    }


# ── Table builder ─────────────────────────────────────────────────────────────

def _build_table(raw_rows: list[str], styles: dict) -> Table | None:
    parsed = []
    for row in raw_rows:
        if re.match(r"^\|\s*[-:]+", row):
            continue
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if cells:
            parsed.append(cells)

    if not parsed:
        return None

    max_cols = max(len(r) for r in parsed)
    data = [r + [""] * (max_cols - len(r)) for r in parsed]

    styled = []
    for i, row in enumerate(data):
        s = styles["thdr"] if i == 0 else styles["tcell"]
        styled.append([Paragraph(cell, s) for cell in row])

    col_width = (6.5 * inch) / max_cols
    tbl = Table(styled, colWidths=[col_width] * max_cols)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#E8E8E8")),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))
    return tbl


# ── Content parser ────────────────────────────────────────────────────────────

def _parse(content: str, styles: dict) -> list:
    flowables = []
    lines = content.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flowables.append(Spacer(1, 5))
            i += 1
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            tbl = _build_table(table_lines, styles)
            if tbl:
                flowables.append(Spacer(1, 4))
                flowables.append(tbl)
                flowables.append(Spacer(1, 6))
            continue

        if re.match(r"^\d{1,2}\s{1,3}\S", stripped):
            flowables.append(Paragraph(stripped, styles["section"]))
            i += 1
            continue

        if stripped.startswith("•") or stripped.startswith("●"):
            text = stripped.lstrip("•● ").strip()
            flowables.append(Paragraph(f"• {text}", styles["bullet"]))
            i += 1
            continue

        flowables.append(Paragraph(stripped, styles["body"]))
        i += 1

    return flowables


# ── PDF renderer ──────────────────────────────────────────────────────────────

def _render_pdf(path: Path, doc_code: str, title: str, content: str) -> None:
    s = _styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    story = [
        Paragraph("Everstorm Outfitters", s["company"]),
        Paragraph(title, s["title"]),
        Paragraph(f"Document {doc_code}", s["code"]),
        HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#AAAAAA")),
        Spacer(1, 10),
    ]

    story.extend(_parse(content, s))
    doc.build(story)


# ── Claude content generation ─────────────────────────────────────────────────

def _generate_content(doc: dict) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": doc["prompt"]}],
    )
    return response.content[0].text


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Generating {len(DOCUMENTS)} PDFs into {DATA_DIR}\n")

    for doc in DOCUMENTS:
        marker = " [CROSS-DOC]" if int(doc["code"].split("-")[1]) >= 13 else ""
        print(f"Generating {doc['filename']}{marker} ...", end=" ", flush=True)
        try:
            content = _generate_content(doc)
            _render_pdf(DATA_DIR / doc["filename"], doc["code"], doc["title"], content)
            print("done")
        except Exception as exc:
            print(f"FAILED — {exc}")

    total = len(list(DATA_DIR.glob("*.pdf")))
    print(f"\n{total} PDFs ready in {DATA_DIR}")
    print("\nCross-document PDFs (GraphRAG entity bridges):")
    for doc in DOCUMENTS:
        if int(doc["code"].split("-")[1]) >= 13:
            print(f"  • {doc['filename']}")


if __name__ == "__main__":
    main()
