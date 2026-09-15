# Manual test: personal avatar phone calls

Use this checklist to walk the LiveKit SIP + LangGraph phone feature the way an owner would. Automated unit tests do not cover a real ring, a real restaurant, or the listen-in bar.

The product uses **one shared platform number**. Standard and Pro never receive a private Neural Nexus number. The owner verifies the **mobile they already have**, calls the shared number from that mobile, and outbound restaurant calls ring that same mobile first so the owner can listen.

## What you need

- A signed-in Neural Nexus account that owns a **personal avatar**
- The **web app** (`f-Neural-Nexus-Frontend`) and the **API** (`f-anubis`) running against the same environment
- The **mobile you will verify** (SMS + voice)
- Optional: an iPhone with `f-anubis-mcp-server-mobile` installed, only for the non-collision check
- Optional: a second person or a second line to play the restaurant, if you do not want to call a real shop

Acceptance fixtures (name + city, not hardcoded vendors):

- **Kanji** in your city
- **Mellow Mushroom** in your city

Payment rule for this version: **pickup / pay at the counter only**. The avatar must never speak a card number.

## 1. Configure the environment

Copy empty keys from `.env.example` into `.env` (production) or `.env.dev` (local compose). Fill the ones you will actually use.

### Required for a real inbound or outbound call

| Variable | Purpose |
|---|---|
| `LIVEKIT_URL` | LiveKit Cloud or self-hosted WebSocket URL |
| `LIVEKIT_API_KEY` | LiveKit API key |
| `LIVEKIT_API_SECRET` | LiveKit API secret |
| `LIVEKIT_SIP_OUTBOUND_TRUNK_ID` | Shared outbound SIP trunk (one trunk for the product) |
| `PLATFORM_PHONE_NUMBER` | The **one shared** inbound DID, E.164 (example: `+18005550199`) |
| `TWILIO_ACCOUNT_SID` | SMS one-time code |
| `TWILIO_AUTH_TOKEN` | SMS one-time code |
| `TWILIO_FROM_NUMBER` | Number Twilio texts from |
| `DEEPGRAM_API_KEY` | Speech-to-text on the phone worker |
| ElevenLabs clone | Already used for avatar TTS; do not stand up ElevenLabs Agents |

### Required for lookup and travel in chat (no SIP needed)

| Variable | Purpose |
|---|---|
| `OPENROUTE_API_KEY` | Driving time. If empty, the call can still proceed and travel fields stay `unknown` |
| `NOMINATIM_BASE_URL` | Leave empty to use `https://nominatim.openstreetmap.org` |
| `OVERPASS_BASE_URL` | Leave empty to use `https://overpass-api.de/api/interpreter` |

`GOOGLE_PLACES_API_KEY` is optional. Leave empty unless OpenStreetMap has no phone for the shop.

### Local development without SMS delivery

Set `PHONE_VERIFY_ALLOW_UNDELIVERED=TRUE` only on a local machine. The API will issue a confirmation code without sending a text. Production must leave this unset or `FALSE`.

When undelivered codes are allowed, read the code from API logs (`Phone verification code for +1… issued without SMS delivery`) or set the code in a debugger. Prefer a real Twilio send for an honest owner test.

### Start the stack

```bash
# API + phone_worker (dev)
docker compose up

# or production compose
./dcrp.sh

# Web app
cd ../f-Neural-Nexus-Frontend && npm install && npm run dev
```

Confirm `phone_worker` is running next to the API. Without `LIVEKIT_URL` the worker stays idle and logs that SIP will not run.

## 2. Lookup and travel — no Phone connection

Do this **before** connecting Phone, and without an iPhone MCP daemon signed in.

1. Sign in and open **your personal avatar** chat (not a shared or visitor avatar).
2. Allow browser location if you want travel time, or be ready to type an origin.
3. Ask: `What is Kanji's phone number in <your city>?`
4. Ask: `How long to Mellow Mushroom in <your city>?`

**Pass**

- The avatar calls `lookup_local_place` and `estimate_travel` (or answers from those tools).
- A real number and address appear, or the avatar says the number was not found. The avatar must not invent a number.
- If no origin is known, the avatar asks where you are. The avatar must not call iOS `get_location`.
- SIP tools are not offered. Asking “call them” should prompt you to **connect Phone**, not place a SIP call.

**Fail**

- Tools missing on the personal avatar.
- Tools appearing on a non-personal avatar.
- A fabricated phone number.
- Any mention of opening the iOS dialer for this lookup.

Repeat the same two questions on a **non-personal** avatar you own. Those tools must stay hidden.

## 3. Connect Phone — verify the mobile you already have

1. On the personal avatar, open **Connections** (or ask the avatar to connect Phone).
2. Choose the **Phone** card: “Use the phone you already have.” This is **not** a device / MCP-connector row.
3. Enter the mobile you already carry. Leave the confirmation-code field blank. Submit.
4. Enter the 6-digit SMS code. Submit again.

**Pass**

- First submit does not create a connection. The card asks for the code.
- After the code, the connection shows the **verified mobile**.
- The card or listing shows the **shared** `PLATFORM_PHONE_NUMBER` to call from that mobile.
- No new Neural Nexus number is assigned. The listing must not show a private `platform_number_e164` of your own.

**Fail**

- A new DID is provisioned or displayed as “your Neural Nexus number.”
- Connect succeeds without a code (unless you are on a local undelivered-code setup and still typed a valid issued code).
- Phone appears under device / iOS MCP connectors.

### 3b. Enterprise-only private number

On a **free or Pro** account, try to request a dedicated inbound number (connect body with `want_private_number=true`, or any UI that asks for a private Neural Nexus number).

**Pass:** HTTP 403 (or a plain on-screen refusal) saying a private number is enterprise-only, and to verify the mobile you already have.

On **premium**, the same request must still **not** buy a number in this slice. Expect “not available yet” (HTTP 501), not a purchased DID.

## 4. Inbound — you call the shared number

1. From the **same mobile you verified**, dial `PLATFORM_PHONE_NUMBER`.
2. Stay on the line.

**Pass**

- LiveKit answers.
- The personal avatar speaks in the **existing ElevenLabs clone** (not a generic ElevenLabs Agent voice).
- The conversation is a normal talk, not a restaurant order script.
- The call is tied to **your** personal avatar (caller ID matched `owner_mobile_e164`).

**Fail**

- Busy signal / immediate hangup when calling from the verified mobile.
- A different avatar answers.
- The product issues you a new number to call instead of the shared one.

### 4b. Unknown caller ID

Call the shared number from a **different** phone that was never verified.

**Pass:** Short reject or a generic “call from the mobile you verified” prompt. No new account, no new number.

## 5. Outbound — order from chat

Do this from the **personal avatar** chat after Phone is connected.

1. Allow location, or say where you are (address or “I am at …”).
2. Send something like:

   `Order a large pepperoni from Mellow Mushroom in <your city> for pickup.`

   or

   `Order tonkotsu from Kanji in <your city> for pickup, name on the order <your name>.`

3. Confirm the interrupt card (**Place the call**). Do not type a card number.
4. Answer your verified mobile when the phone rings. You are a **listener** on the restaurant call.
5. If you miss the ring, watch the **web listen-in bar** on the personal-avatar chat (muted by default).
6. After hangup, open **chat** and the **agent inbox**.

**Pass**

- A confirm card appears **before** any ring. Cancelling the card places no call.
- Your verified mobile rings **first**, then the restaurant.
- The avatar places a pickup order and does **not** ask for or speak a card number.
- If the restaurant demands a card on the line, the call ends and the inbox says so.
- Chat and inbox both show **cost**, **ready time**, **address**, and **travel time** (travel may be `unknown` if `OPENROUTE_API_KEY` is empty).
- Inbox item source is a phone-call notification (acknowledge only; no “reply to the restaurant”).

**Fail**

- Call starts without the confirm card.
- Restaurant rings before your mobile.
- iOS `place_call` / the system dialer opens instead of LiveKit SIP.
- Card number is spoken.
- Result appears in only chat or only inbox, or invents cost / time / address.

### 5b. Confirm cancelled

Start the same order and press **Not now** on the confirm card.

**Pass:** No ring, no `phone_calls` row in progress, avatar says the call was not placed.

## 6. Web listen-in (fallback)

1. Start an outbound order.
2. Do **not** answer the owner mobile.
3. Stay on the personal-avatar chat page.

**Pass**

- A compact listen bar appears (`Ringing your mobile` then `Listening`).
- The bar is muted / subscribe-only. You hear the room; your browser mic stays off.
- When the call ends, the bar goes away and the result still lands in chat + inbox.

If LiveKit keys are missing, the bar may show that web listen-in is unavailable. The restaurant leg should still have been attempted.

## 7. iOS `place_call` must not change

On an iPhone with the Neural Nexus MCP app connected:

1. Ask the personal avatar to **dial a number on this iPhone** (the existing mobile tool).
2. Confirm the iOS prompt.

**Pass**

- iOS opens the **system dialer** (`tel://`).
- No CallKit, VoIP, or SIP inside the iOS app.
- The SIP tools in chat stay named `request_outbound_phone_call`, `get_phone_call_status`, `end_phone_call` — never `place_call`.

After a SIP restaurant call finishes, an optional local notification is allowed if an iPhone is bound. That notification is not a substitute for the inbox + chat result.

## 8. Surfaces and gates (quick pass)

| Surface | Expected |
|---|---|
| Personal avatar, Phone **not** connected | Lookup + travel yes; SIP tools no |
| Personal avatar, Phone connected | Lookup + travel + SIP tools |
| Shared / visitor avatar | No lookup, travel, or SIP tools |
| Connections picker | Phone card with a phone icon, category Phone |
| `GET /connectable_providers` | Includes `phone` |
| `POST /mcp/phone` with your API key | `tools/list` includes lookup + travel; SIP names only after Phone is connected; no `place_call` |
| `POST /phone_calls/{id}/listen` | Subscribe-only LiveKit token for a call you own |

## Suggested script (happy path, ~20 minutes)

1. Personal avatar, no Phone: lookup Kanji, estimate travel to Mellow Mushroom.
2. Connect Phone; verify your mobile; note the shared inbound number.
3. Call that shared number from the verified mobile; talk for 20–30 seconds; hang up.
4. In chat, order a named item from Kanji or Mellow Mushroom; confirm; answer your mobile; listen to the restaurant leg.
5. Check chat + inbox for cost, ready time, address, travel time.
6. Repeat the order and cancel the confirm card once.
7. If you have iOS MCP: trigger `place_call` and confirm only the system dialer opens.

## When something fails

- **No SMS code:** Twilio env vars, or `PHONE_VERIFY_ALLOW_UNDELIVERED` on local only.
- **Lookup has no phone:** OSM may lack a `phone` tag. Optional `GOOGLE_PLACES_API_KEY`, or pick the other fixture. The avatar must refuse to dial, not guess.
- **Travel is unknown:** missing `OPENROUTE_API_KEY`, or no origin (enable browser location or say where you are).
- **No ring:** `phone_worker`, LiveKit URL/keys, `LIVEKIT_SIP_OUTBOUND_TRUNK_ID`, and `PLATFORM_PHONE_NUMBER` on the same environment the API uses.
- **Wrong avatar on inbound:** you called from a number that is not the verified `owner_mobile_e164`.
- **Listen bar silent:** browser blocked the LiveKit connection, or listen token failed (503 if LiveKit is not configured).
- **Worker idle:** `LIVEKIT_URL` empty; the worker logs that SIP will stay idle.

Do not test card-over-phone, per-user number purchase on Standard/Pro, CallKit in the iOS app, or ElevenLabs Agents. Those are out of scope on purpose.
