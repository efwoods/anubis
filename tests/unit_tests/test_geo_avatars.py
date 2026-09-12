"""Geo-located avatars: an avatar pinned to a real-world place.

An avatar may be pinned to a memorial, a grave marker, a monument, an exhibit or
a storefront. The pin puts the avatar on the world map, lets a passer-by be told
that an avatar stands here, and changes how the avatar speaks to someone who is
standing at the place. Nobody is ever refused a conversation for being somewhere
else: presence is a trigger, never a permission check.

These tests cover the pure geometry and validation, the two throttles, the rule
that the pin stays public while the rest of an avatar's metadata does not, and
the ownership gate on moving an avatar in the physical world.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.anubis.utils.geo import (
    CheckinThrottle,
    GeoLocationError,
    avatars_near,
    build_geo_location,
    geo_location_of,
    haversine_distance_meters,
    render_avatar_place_section,
    validate_coordinates,
    validate_geofence_radius,
    within_bounds,
)
from src.api import webapp as webapp_module

# Minneapolis Stone Arch Bridge and a point about 120 m east of the bridge.
BRIDGE = (44.9809, -93.2533)
NEAR_BRIDGE = (44.9809, -93.2518)
# Saint Paul cathedral, about 14 km away.
CATHEDRAL = (44.9469, -93.1089)

ASSISTANT_ID = "assistant-alpha"
CREATOR_ID = "6a5e59310832afadd626e583"
STRANGER_ID = "someone-else"
ADMIN_ID = "the-admin"


def _current_user(user_id):
    return {"API_KEY": "sk-test-key", "identities": [{"user_id": user_id}]}


def _avatar(assistant_id, point, *, radius=50, public=True, name=None):
    return {
        "assistant_id": assistant_id,
        "name": name or assistant_id,
        "description": "a place in the world",
        "metadata": {
            "is_public": public,
            "user_id": CREATOR_ID,
            "geo_location": build_geo_location(*point, geofence_radius_meters=radius),
        },
    }


""" Coordinates, radii, and the stored pin """


def test_validate_coordinates_and_radius():
    assert validate_coordinates("44.98", "-93.25") == (44.98, -93.25)
    for rejected in ((91, 0), (0, 181), ("x", 0), (float("nan"), 0)):
        with pytest.raises(GeoLocationError):
            validate_coordinates(*rejected)
    assert validate_geofence_radius(None) == 50
    assert validate_geofence_radius("120") == 120
    assert validate_geofence_radius(1) == 1
    assert validate_geofence_radius("1") == 1
    for rejected in (0, 0.4, 999999, "wide"):
        with pytest.raises(GeoLocationError):
            validate_geofence_radius(rejected)


def test_build_geo_location_block():
    block = build_geo_location(
        *BRIDGE, location_name="  Stone Arch Bridge ", geofence_radius_meters=80
    )
    assert block["latitude"] == BRIDGE[0]
    assert block["location_name"] == "Stone Arch Bridge"
    assert block["geofence_radius_meters"] == 80
    assert block["geo_located_at"]
    # The pin reads the same from the owner's metadata and from a public listing.
    assert geo_location_of({"metadata": {"geo_location": block}}) == block
    assert geo_location_of({"geo_location": block}) == block
    assert (
        geo_location_of({"metadata": {"geo_location": {"latitude": 999, "longitude": 0}}})
        is None
    )
    assert geo_location_of({"metadata": {}}) is None


def test_haversine_distance_is_accurate():
    assert haversine_distance_meters(*BRIDGE, *BRIDGE) == 0
    assert 100 < haversine_distance_meters(*BRIDGE, *NEAR_BRIDGE) < 140
    assert 11_000 < haversine_distance_meters(*BRIDGE, *CATHEDRAL) < 12_500


""" Finding the avatars around a person """


def test_avatars_near_ranks_and_reports_geofences():
    avatars = [
        _avatar("cathedral", CATHEDRAL, radius=200),
        _avatar("bridge", BRIDGE, radius=150),
        {"assistant_id": "unpinned", "name": "no place", "metadata": {"is_public": True}},
        _avatar("bridge-tight", BRIDGE, radius=20, public=False),
    ]
    nearby = avatars_near(*NEAR_BRIDGE, avatars, radius_meters=500)
    assert [entry["assistant_id"] for entry in nearby] == ["bridge", "bridge-tight"]
    assert nearby[0]["inside_geofence"] is True
    assert nearby[1]["inside_geofence"] is False
    # A phone that is only sure of the position to within 120 m widens the fence,
    # so a visitor genuinely standing at the place is not told they are outside.
    widened = avatars_near(
        *NEAR_BRIDGE, avatars, radius_meters=500, accuracy_meters=120
    )
    assert widened[1]["inside_geofence"] is True
    everything = avatars_near(*NEAR_BRIDGE, avatars, radius_meters=20_000)
    assert [entry["assistant_id"] for entry in everything] == [
        "bridge",
        "bridge-tight",
        "cathedral",
    ]
    assert everything[-1]["distance_meters"] > 11_000


def test_within_bounds_handles_open_sides_and_the_antimeridian():
    block = build_geo_location(*BRIDGE)
    assert within_bounds(
        block,
        min_latitude=44.9,
        min_longitude=-93.3,
        max_latitude=45.0,
        max_longitude=-93.2,
    )
    assert not within_bounds(
        block,
        min_latitude=45.1,
        min_longitude=None,
        max_latitude=None,
        max_longitude=None,
    )
    assert within_bounds(
        block,
        min_latitude=None,
        min_longitude=None,
        max_latitude=None,
        max_longitude=None,
    )
    # A map viewport dragged across the antimeridian wraps.
    fiji = build_geo_location(-17.7, 178.0)
    assert within_bounds(
        fiji, min_latitude=-30, min_longitude=170, max_latitude=0, max_longitude=-170
    )
    assert not within_bounds(
        fiji, min_latitude=-30, min_longitude=-170, max_latitude=0, max_longitude=170
    )
    assert not within_bounds(
        None,
        min_latitude=None,
        min_longitude=None,
        max_latitude=None,
        max_longitude=None,
    )


""" Throttles: a moving phone must not flood visits or notifications """


def test_checkin_throttle_limits_visits_per_visitor_and_avatar():
    throttle = CheckinThrottle(min_interval_seconds=300)
    assert throttle.allow("visitor", "bridge", now=0.0) is True
    assert throttle.allow("visitor", "bridge", now=100.0) is False
    assert throttle.allow("visitor", "cathedral", now=100.0) is True
    assert throttle.allow("other", "bridge", now=100.0) is True
    assert throttle.allow("visitor", "bridge", now=301.0) is True


def test_notification_cooldown_is_a_second_throttle():
    notify = CheckinThrottle(min_interval_seconds=3600)
    assert notify.allow("visitor", "bridge", now=0.0) is True
    assert notify.allow("visitor", "bridge", now=1800.0) is False
    assert notify.allow("visitor", "bridge", now=3601.0) is True


""" The place in the system prompt """


def test_render_avatar_place_section():
    block = build_geo_location(*BRIDGE, location_name="Stone Arch Bridge")
    away = render_avatar_place_section(block, visitor_present=False)
    assert "Stone Arch Bridge" in away
    assert "not at that place" in away
    here = render_avatar_place_section(block, visitor_present=True)
    assert "standing at that place right now" in here
    assert render_avatar_place_section(None) == ""


def test_the_prompt_template_has_a_place_slot():
    """The avatar's place reaches the prompt, and renders empty when unpinned."""
    from src.anubis.utils.classes.DynamicPromptBuilder import DynamicPromptBuilder

    builder = DynamicPromptBuilder()
    pinned = builder.build_prompt(
        assistant_name="Marine",
        assistant_place=render_avatar_place_section(
            build_geo_location(*BRIDGE, location_name="Stone Arch Bridge"),
            visitor_present=True,
        ),
    ).messages[0].content
    assert "=== YOUR PLACE ===" in pinned
    assert "Stone Arch Bridge" in pinned

    unpinned = builder.build_prompt(assistant_name="Marine").messages[0].content
    assert "=== YOUR PLACE ===\n\n" in unpinned
    assert "Stone Arch Bridge" not in unpinned


""" The pin is public; the rest of the metadata is not """


def test_public_listing_keeps_the_pin_but_not_the_metadata():
    listed = webapp_module._assistant_without_metadata_if_public(_avatar("bridge", BRIDGE))
    assert "metadata" not in listed
    assert listed["geo_location"]["latitude"] == BRIDGE[0]
    # The creator still sees the whole record, which is how the client decides
    # whether to offer the Avatar Settings tab.
    own = webapp_module._assistant_without_metadata_if_public(
        _avatar("bridge", BRIDGE), viewer_user_id=CREATOR_ID
    )
    assert "metadata" in own


def test_stripping_metadata_never_leaks_the_creator_identifier():
    stripped = webapp_module._assistant_without_metadata(_avatar("bridge", BRIDGE))
    assert "metadata" not in stripped
    assert CREATOR_ID not in str(stripped)
    assert stripped["geo_location"]["geofence_radius_meters"] == 50
    # An avatar with no pin simply loses the metadata.
    unpinned = webapp_module._assistant_without_metadata(
        {"assistant_id": "x", "metadata": {"user_id": CREATOR_ID}}
    )
    assert "geo_location" not in unpinned


""" Validating a pin on the way in """


def test_half_a_coordinate_pair_is_refused():
    with pytest.raises(webapp_module.HTTPException) as refused:
        webapp_module._build_geo_location_or_400(
            latitude=44.9809,
            longitude=None,
            location_name=None,
            geofence_radius_meters=None,
        )
    assert refused.value.status_code == 400
    assert "both latitude and longitude" in refused.value.detail

    with pytest.raises(webapp_module.HTTPException) as out_of_range:
        webapp_module._build_geo_location_or_400(
            latitude=91.0,
            longitude=0.0,
            location_name=None,
            geofence_radius_meters=None,
        )
    assert out_of_range.value.status_code == 400

    assert (
        webapp_module._build_geo_location_or_400(
            latitude=None,
            longitude=None,
            location_name=None,
            geofence_radius_meters=None,
        )
        is None
    )


""" Moving an avatar in the world is the creator's call """


class _AssistantsAPI:
    def __init__(self, metadata):
        self._metadata = metadata
        self.updates = []

    async def get(self, assistant_id):
        return {"assistant_id": assistant_id, "metadata": self._metadata}

    async def update(self, **kwargs):
        self.updates.append(kwargs)
        return {"assistant_id": kwargs.get("assistant_id")}


def _install(monkeypatch, metadata):
    assistants_api = _AssistantsAPI(metadata)
    monkeypatch.setattr(
        webapp_module,
        "get_client",
        lambda **kwargs: SimpleNamespace(assistants=assistants_api),
    )
    monkeypatch.setattr(
        webapp_module.app.state,
        "context",
        SimpleNamespace(admin_user_id=ADMIN_ID),
        raising=False,
    )
    return assistants_api


def _request_without_query_parameters():
    return SimpleNamespace(query_params={})


@pytest.mark.asyncio
async def test_the_creator_may_pin_move_and_clear_the_avatar(monkeypatch):
    assistants_api = _install(monkeypatch, {"user_id": CREATOR_ID})

    await webapp_module.modify_avatar(
        request=_request_without_query_parameters(),
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(CREATOR_ID),
        latitude=BRIDGE[0],
        longitude=BRIDGE[1],
        location_name="Stone Arch Bridge",
        geofence_radius_meters=120,
    )
    pinned = assistants_api.updates[-1]["metadata"]["geo_location"]
    assert pinned["location_name"] == "Stone Arch Bridge"
    assert pinned["geofence_radius_meters"] == 120
    # The pin is the only metadata key written, so user_id and is_public survive
    # the merge.
    assert set(assistants_api.updates[-1]["metadata"]) == {"geo_location"}

    await webapp_module.modify_avatar(
        request=_request_without_query_parameters(),
        assistant_id=ASSISTANT_ID,
        current_user=_current_user(CREATOR_ID),
        clear_geo_location=True,
    )
    assert assistants_api.updates[-1]["metadata"]["geo_location"] is None


@pytest.mark.asyncio
async def test_a_stranger_may_not_move_the_avatar(monkeypatch):
    assistants_api = _install(monkeypatch, {"user_id": CREATOR_ID})

    with pytest.raises(webapp_module.HTTPException) as refused:
        await webapp_module.modify_avatar(
            request=_request_without_query_parameters(),
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(STRANGER_ID),
            latitude=CATHEDRAL[0],
            longitude=CATHEDRAL[1],
        )
    assert refused.value.status_code == 403
    assert assistants_api.updates == []


@pytest.mark.asyncio
async def test_modify_avatar_still_refuses_an_empty_request(monkeypatch):
    _install(monkeypatch, {"user_id": CREATOR_ID})

    with pytest.raises(webapp_module.HTTPException) as refused:
        await webapp_module.modify_avatar(
            request=_request_without_query_parameters(),
            assistant_id=ASSISTANT_ID,
            current_user=_current_user(CREATOR_ID),
        )
    assert refused.value.status_code == 400
    assert "clear_geo_location" in refused.value.detail


""" The routes the web application calls """


def _query_parameter_names(path, method):
    for route in webapp_module.app.routes:
        if getattr(route, "path", None) == path and method in getattr(
            route, "methods", ()
        ):
            return {
                parameter.name: parameter.field_info.is_required()
                for parameter in route.dependant.query_params
            }
    raise AssertionError(f"route not registered: {method} {path}")


def test_the_geo_routes_are_registered_with_the_expected_parameters():
    map_listing = _query_parameter_names("/avatars/geo", "GET")
    for side in ("min_latitude", "min_longitude", "max_latitude", "max_longitude"):
        assert map_listing.get(side) is False

    nearby = _query_parameter_names("/avatars/nearby", "GET")
    assert nearby.get("latitude") is True
    assert nearby.get("longitude") is True
    assert nearby.get("radius_meters") is False
    assert nearby.get("accuracy_meters") is False

    registered_paths = {
        getattr(route, "path", None) for route in webapp_module.app.routes
    }
    assert "/geo/checkin" in registered_paths

    creation = _query_parameter_names("/create_avatar", "POST")
    modification = _query_parameter_names("/modify_avatar", "PATCH")
    for pin_parameter in (
        "latitude",
        "longitude",
        "location_name",
        "geofence_radius_meters",
    ):
        assert creation.get(pin_parameter) is False
        assert modification.get(pin_parameter) is False
    assert modification.get("clear_geo_location") is False


def test_a_message_may_say_the_person_is_standing_at_the_place():
    """``at_place`` rides the message request and is never a permission check."""
    for route in webapp_module.app.routes:
        if getattr(route, "path", None) == "/message/{assistant_id}":
            body_fields = {
                parameter.name for parameter in route.dependant.body_params
            }
            assert "at_place" in body_fields
            return
    raise AssertionError("route not registered: POST /message/{assistant_id}")


""" Configuration """


def test_the_geo_limits_come_from_the_environment(monkeypatch):
    from src.anubis.utils.context import GlobalContext

    for variable in (
        "GEO_CHECKIN_MIN_INTERVAL_SECONDS",
        "GEO_NEARBY_DEFAULT_RADIUS_METERS",
        "GEO_NEARBY_MAX_RADIUS_METERS",
        "GEO_NOTIFY_COOLDOWN_SECONDS",
    ):
        monkeypatch.delenv(variable, raising=False)
    defaults = GlobalContext()
    assert defaults.geo_checkin_min_interval_seconds == 300
    assert defaults.geo_nearby_default_radius_meters == 500
    assert defaults.geo_nearby_max_radius_meters == 50_000
    assert defaults.geo_notify_cooldown_seconds == 3600

    monkeypatch.setenv("GEO_NEARBY_DEFAULT_RADIUS_METERS", "750")
    monkeypatch.setenv("GEO_NOTIFY_COOLDOWN_SECONDS", "60")
    configured = GlobalContext()
    assert configured.geo_nearby_default_radius_meters == 750
    assert configured.geo_notify_cooldown_seconds == 60
