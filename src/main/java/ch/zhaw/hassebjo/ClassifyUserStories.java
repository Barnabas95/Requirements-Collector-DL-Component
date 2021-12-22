package ch.zhaw.hassebjo;

import ch.zhaw.hassebjo.model.UserStoryContainer;

import java.io.IOException;

public interface ClassifyUserStories {

    /**
     * @param projectId the projectId used in the Storyscreen webapplication
     * @param inputBytes the file sent via REST to be evaluated
     * @return {@link UserStoryContainer} with all found user stories
     * @throws IOException
     */
    public UserStoryContainer processFile(Integer projectId, byte[] inputBytes) throws IOException;

    /**
     * @param inputBytes the file sent via REST to be used to train the model
     * @throws IOException
     */
    public void trainModel(byte[] inputBytes) throws IOException;
}
