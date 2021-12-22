package ch.zhaw.hassebjo;

import ch.zhaw.hassebjo.model.UserStoryContainer;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

@RestController
public class ClassifyController {

    private final ClassifyUserStories classifyUserStories;

    public ClassifyController(ClassifyUserStories classifyUserStories) {
        this.classifyUserStories = classifyUserStories;
    }

    @PostMapping("classify")
    public ResponseEntity<UserStoryContainer> uploadFile(@RequestPart("file") MultipartFile file,
                                                         @RequestParam("projectId") Integer projectId) {
        if (null == file.getOriginalFilename()) {
            return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
        }
        UserStoryContainer userStoryContainer = null;
        try {
            byte[] bytes = file.getBytes();

            System.out.println("File received - Beginning processing");
            userStoryContainer = classifyUserStories.processFile(projectId, file.getBytes());

        } catch (IOException e) {
            System.out.println(e.getMessage());
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
        return new ResponseEntity<>(userStoryContainer, HttpStatus.OK);
    }

    @PostMapping("train")
    public ResponseEntity<UserStoryContainer> trainModel(@RequestPart("file") MultipartFile file) {
        if (null == file.getOriginalFilename()) {
            return new ResponseEntity<>(HttpStatus.BAD_REQUEST);
        }
        try {
            byte[] bytes = file.getBytes();

            System.out.println("File received - Beginning training");
            classifyUserStories.trainModel(file.getBytes());


        } catch (IOException e) {
            System.out.println(e.getMessage());
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
        return new ResponseEntity<>(HttpStatus.OK);
    }
}
