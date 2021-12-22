package ch.zhaw.hassebjo.model;

import java.util.ArrayList;
import java.util.List;

public class UserStoryContainer {

    private List<UserStory> userStories = new ArrayList<>();

    public List<UserStory> getUserStories() {
        return userStories;
    }

    public void setUserStories(List<UserStory> userStories) {
        this.userStories = userStories;
    }

    public void addUserStory(UserStory userStory) {
        userStories.add(userStory);
    }
}
