I still want other users to log in however I only want the original creator to be able to talk to their own created avatars as well as public avatars. I need to be able to allow for avatars to be shared if "switched" to public and I need the list of avatars to include all shared avatars and avatars created by the user. Users that did not create an avatar cannot delete that avatar for example. I need the langgraph_auth_user object to allow users to interact with a place for prayers avatar which is created and shared with the public with an anonymous sign in. I need to verify email accounts after signup

how i save 30 second of recorded audio or an uploaded reference audio, how I save a user profile reference image and description on signup or on a separate add reference user image and add reference user audio routes 

there is a picture under Identity provider attributes:
https://s.gravatar.com/avatar/90369d12b0b93a93a3228238b1d50a11?s=480&r=pg&d=https%3A%2F%2Fcdn.auth0.com%2Favatars%2Fev.png

how do I require email verification after signup?


there are other routes that are created on default that I do not define. How do I apply authentication logic to these routes as described above?


on create if I create the avatar as public or supply a route to make the avatar public after creation then I would check the metadata for public or shared and allow the avatar to be shared with the public regardless of creation. How do I create anonymous default users for anyone to communicate with a single public avatar that I created (place for prayers) without needing to sign up as a convenience with a rate limit for messaging and a prompt to signup after n messages or n tokens? I need to use python stripe.

how do I store and create urls for reference images and reference audio? how do I allow a user to record 30 seconds of audio with python from an api endpoint? 


Create a reusable security dependency
async def validate_avatar_owner(
    avatar_id: str, 
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["sub"]
    is_owner = await check_db_ownership(avatar_id, user_id)
    
    if not is_owner:
        raise HTTPException(status_code=403, detail="Access Denied")
    return avatar_id

# Use it in any route
@security_route.delete("/avatars/{avatar_id}")
async def delete_avatar(valid_id: str = Depends(validate_avatar_owner)):
    # This code ONLY runs if the ownership check passed
    await db.delete_avatar(valid_id)
    return {"message": "Deleted"}