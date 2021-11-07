package ch.zhaw.widmemor;

import ch.zhaw.widmemor.model.UserStoryContainer;

import java.io.IOException;

public interface ClassifyUserStories {

    //tmp void until output has been formatted
    public UserStoryContainer processFile(Integer projectId, byte[] inputBytes) throws IOException;

}
