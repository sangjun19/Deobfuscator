// Repository: pradeepnarava/tfh_latest
// File: BeteendeexperimentController.m

//
//  BeteendeexperimentController.m
//  Välkommen till TFH-appen
//
//  Created by Mohammed Abdul Majeed on 5/2/13.
//  Copyright (c) 2013 brilliance. All rights reserved.
//

#import "BeteendeexperimentController.h"
#import "MTPopupWindow.h"


int c=0;


#define kAlertViewOne 1
#define kAlertViewTwo 2

@interface BeteendeexperimentController ()

@property (nonatomic) BOOL isSaved;

@end

@implementation BeteendeexperimentController
@synthesize label1,ex3c1,ex3c2,ex3c3,ex3c4,ex3c5,slabel1,slabel2,tableview,listexercise4,list_exercise4;
@synthesize isSaved;



- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
    self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
    if (self) {
        // Custom initialization

    }
    return self;
}



#pragma  mark TextViewDelegate Methods
-(BOOL)textView:(UITextView *)textView shouldChangeTextInRange:(NSRange)range replacementText:(NSString *)text
{
    isSaved = YES;
    if([text isEqualToString:@"\n"]) {
        [textView resignFirstResponder];
        if (textView == ex3c5) {
            
            if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone) {
                if ([[UIScreen mainScreen] bounds].size.height >  480 ) {
        [UIView animateWithDuration:0.5
                         animations:^{
                              [scroll setContentOffset:CGPointMake(scroll.frame.origin.x, scroll.frame.origin.y + 753) animated:YES];
                         }
                         completion:^(BOOL finished){
                             // whatever you need to do when animations are complete
                             
                         }];
                }
                else {
                    [UIView animateWithDuration:0.5
                                     animations:^{
                                         [scroll setContentOffset:CGPointMake(scroll.frame.origin.x, scroll.frame.origin.y + 840) animated:YES];
                                     }
                                     completion:^(BOOL finished){
                                         // whatever you need to do when animations are complete
                                         
                                     }];
                }
            }
        }
    }
    else {
        return YES;
    }
    return 0;
}

- (BOOL)textViewShouldBeginEditing:(UITextView *)textView
{
    [UIView animateWithDuration:0.5
                     animations:^{
                         if (textView == ex3c2) {
                             NSLog(@"1");
                             [scroll setContentOffset:CGPointMake(scroll.frame.origin.x, ex3c2.frame.origin.y - 30) animated:YES];
                         }
                         if (textView == ex3c3) {
                             NSLog(@"2");
                             [scroll setContentOffset:CGPointMake(scroll.frame.origin.x, ex3c3.frame.origin.y - 30) animated:YES];
                         }
                         if (textView == ex3c4) {
                             NSLog(@"3");
                             [scroll setContentOffset:CGPointMake(scroll.frame.origin.x, ex3c4.frame.origin.y - 30) animated:YES];
                         }
                         if (textView == ex3c5) {
                             NSLog(@"4");
                             [scroll setContentOffset:CGPointMake(scroll.frame.origin.x, ex3c5.frame.origin.y - 30) animated:YES];
                         }
                         
                     }
                     completion:^(BOOL finished){
                         // whatever you need to do when animations are complete
                         
                     }];
    
    return YES;
}


-(BOOL)textFieldShouldReturn:(UITextField *)textField {
    
//    [ex3c1 resignFirstResponder];
    picker.hidden=YES;
    return YES;
}

- (void)textFieldDidBeginEditing:(UITextField *)aTextField{
    [ex3c1 resignFirstResponder];
    
    picker.hidden=NO;
    [picker addTarget:self action:@selector(dueDateChanged:) forControlEvents:UIControlEventValueChanged];
    


}

#pragma mark ViewLife Cycle 

- (void)viewDidLoad
{
    self.navigationItem.title=@"Beteendeexperiment";
    
    scroll.tag =0;
    scroll1.tag = 1;
    
    
    if ([[UIDevice currentDevice] userInterfaceIdiom] == UIUserInterfaceIdiomPhone) {
        
        UIImage *image = [UIImage imageNamed:@"tillbaka1.png"];
        UIButton *okBtn = [UIButton buttonWithType:UIButtonTypeCustom];
        [okBtn setFrame:CGRectMake(0, 0, image.size.width, image.size.height)];
        [okBtn setBackgroundImage:image forState:UIControlStateNormal];
        [okBtn addTarget:self action:@selector(backButon) forControlEvents:UIControlEventTouchUpInside];
        self.navigationItem.leftBarButtonItem =  [[UIBarButtonItem alloc] initWithCustomView:okBtn];
        
    }
    else {
        
        UIImage *image = [UIImage imageNamed:@"tillbaka1.png"];
        UIButton *okBtn = [UIButton buttonWithType:UIButtonTypeCustom];
        [okBtn setFrame:CGRectMake(0, 0, image.size.width, image.size.height)];
        [okBtn setBackgroundImage:image forState:UIControlStateNormal];
        [okBtn addTarget:self action:@selector(backButon) forControlEvents:UIControlEventTouchUpInside];
        self.navigationItem.leftBarButtonItem =  [[UIBarButtonItem alloc] initWithCustomView:okBtn];
    }
    
    
    [self.view addSubview:listofdates];
    picker.hidden=YES;
    raderabutton.enabled=NO;
    listofdates.hidden=YES;
    questionView3.hidden =YES;
    scroll.scrollEnabled = YES;
    scroll1.scrollEnabled = YES;
    [scroll1 setContentSize:CGSizeMake(768, 1298)];
    [scroll setContentSize:CGSizeMake(320, 1253)];
     list_exercise4=[[NSMutableArray alloc]init];
    [list_exercise4 addObject:@"Null"];
    
    label1.userInteractionEnabled = YES;
    
    
    UITapGestureRecognizer *tapGesture2 =
    [[[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(l1alert:)] autorelease];
    [label1 addGestureRecognizer:tapGesture2];
    
    
    
    
    NSString *docsDir;
    NSArray *dirPaths;
    
    // Get the documents directory
    dirPaths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    
    docsDir = [dirPaths objectAtIndex:0];
    
    // Build the path to the database file
    databasePath = [[NSString alloc] initWithString: [docsDir stringByAppendingPathComponent: @"exerciseDB.db"]];
    
   		const char *dbpath = [databasePath UTF8String];
        
        if (sqlite3_open(dbpath, &exerciseDB) == SQLITE_OK)
        {
            char *errMsg;
            const char *sql_stmt = "CREATE TABLE IF NOT EXISTS EXERCISE4 (ID INTEGER PRIMARY KEY AUTOINCREMENT, DATE TEXT,  DATUM TEXT ,EXPERIMENTET TEXT,FORUTSAGE TEXT,FORUTPRC TEXT, RESULTAT TEXT,LARDOMAR TEXT,LARDPRC TEXT)";
            
            if (sqlite3_exec(exerciseDB, sql_stmt, NULL, NULL, &errMsg) != SQLITE_OK)
            {
                NSLog(@"Failed to create database");
            }else{
                 NSLog(@"create database");
            }
            
            
            [self LoadSavedData];
            sqlite3_close(exerciseDB);
            
        } else {
            //status.text = @"Failed to open/create database";
        }

    
    [super viewDidLoad];
    // Do any additional setup after loading the view from its nib.
}



-(void)backButon {
    
    [self.navigationController popViewControllerAnimated:YES];
}






-(void)viewWillAppear:(BOOL)animated {
    [super viewWillAppear:animated];
    
}



-(void) dueDateChanged:(UIDatePicker *)sender {
    NSDateFormatter* dateFormatter = [[[NSDateFormatter alloc] init] autorelease];
    [dateFormatter setDateStyle:NSDateFormatterLongStyle];
    [dateFormatter setTimeStyle:NSDateFormatterNoStyle];
    picker.hidden=YES;
    //self.myLabel.text = [dateFormatter stringFromDate:[dueDatePickerView date]];
    NSLog(@"Picked the date %@", [dateFormatter stringFromDate:[sender date]]);
}


-(IBAction)mainlabelalert:(id)sender{
     [MTPopupWindow showWindowWithHTMLFile:@"Beteebdeexperiment.html" insideView:self.view];
}


-(void)l1alert:(id)sender{
//     [MTPopupWindow showWindowWithHTMLFile:@"tanke.html" insideView:self.view];
    questionView3.hidden = NO;
}

-(IBAction)questionCloseBtn:(id)sender{
    questionView3.hidden = YES;
}


-(IBAction)changeSlider:(id)sender {
    
    NSString *sl1= [[NSString alloc] initWithFormat:@"%d%@", (int)slider.value,@"%"];
    NSLog(@"sl1%@",sl1);
    self.slabel1.text=sl1;
     NSLog(@"self.slabel1.text%@",self.slabel1.text);
    
}
-(IBAction)changeSlider1:(id)sender {
    
    NSString *sl2= [[NSString alloc] initWithFormat:@"%d%@", (int)slider1.value,@"%"];
    NSLog(@"str%@",sl2);
    self.slabel2.text=sl2;
     NSLog(@"self.slabel2.text%@",self.slabel2.text);
    
}


-(IBAction)saveButton:(id)sender {
    NSDate* date = [NSDate date];
    
    //Create the dateformatter object
    
    NSDateFormatter* formatter = [[[NSDateFormatter alloc] init] autorelease];
    
    //Set the required date format
    
    [formatter setDateFormat:@"MMM d YYYY HH:mm:ss"];
    
    //Get the string date
    
    NSString* str = [formatter stringFromDate:date];
    
    NSLog(@"date%@",str);
    raderabutton.enabled=NO;
    if ([ex3c1.text isEqualToString:@""] &&[ex3c2.text isEqualToString:@""]&&[ex3c3.text isEqualToString:@""]&&[ex3c4.text isEqualToString:@""]&&[ex3c5.text isEqualToString:@""]) {
        
    }else{
        raderabutton.enabled=YES;
        //DATE TEXT,  DATUM TEXT ,EXPERIMENTET TEXT,FORUTSAGE TEXT, RESULTAT TEXT,LARDOMAR TEXT
        const char *dbpath = [databasePath UTF8String];
        
        if (sqlite3_open(dbpath, &exerciseDB) == SQLITE_OK)
        {
            if([[list_exercise4 objectAtIndex:0]  isEqualToString:@"Null"]){
                NSString *insertSQL = [NSString stringWithFormat: @"INSERT INTO EXERCISE4 (date,datum,experimentet,forutsage,forutprc,resultat,lardomar,lardprc) VALUES (\"%@\", \"%@\", \"%@\" ,\"%@\",\"%@\",\"%@\",\"%@\",\"%@\")", str, ex3c1.text,ex3c2.text, ex3c3.text , slabel1.text, ex3c4.text,ex3c5.text,slabel2.text];
                
                const char *insert_stmt = [insertSQL UTF8String];
                
                
                sqlite3_prepare_v2(exerciseDB, insert_stmt, -1, &statement, NULL);
                if (sqlite3_step(statement) == SQLITE_DONE)
                {
                    
                    //isSaved = NO;
                
                } else {
                    NSLog(@"no");
                }
                sqlite3_finalize(statement);
                sqlite3_close(exerciseDB);
            }else{
                NSString *query=[NSString stringWithFormat:@"UPDATE EXERCISE4 SET  datum='%@',experimentet='%@', forutsage='%@',forutprc='%@', resultat='%@',lardomar='%@' ,lardprc='%@' WHERE date='%@'",ex3c1.text,ex3c2.text, ex3c3.text,slabel1.text, ex3c4.text, ex3c5.text,slabel2.text, [listexercise4 objectAtIndex:c] ];
                const char *del_stmt = [query UTF8String];
                
                if (sqlite3_prepare_v2(exerciseDB, del_stmt, -1, & statement, NULL)==SQLITE_OK);{
                    if(SQLITE_DONE != sqlite3_step(statement))
                        NSLog(@"Error while updating. %s", sqlite3_errmsg(exerciseDB));
                    NSLog(@"sss");
                    //isSaved = NO;
        
                }
            
                sqlite3_finalize(statement);
                sqlite3_close(exerciseDB);
            }
        }
        UIAlertView * alert1 = [[UIAlertView alloc] initWithTitle:nil message:@"Sparat" delegate:nil cancelButtonTitle:nil otherButtonTitles:@"Ok",nil];
        [alert1 show];
        [alert1 release];
    }
}


-(IBAction)newButton:(id)sender {
    if([ex3c1.text isEqualToString:@""] && [ex3c2.text isEqualToString:@""] && [ex3c3.text isEqualToString:@""] && [ex3c4.text isEqualToString:@""]&& [ex3c5.text isEqualToString:@""]){
        
    }else{
        
        if (isSaved == YES) {
            UIAlertView  *alert=[[UIAlertView alloc] initWithTitle:nil message:@"Vill du ta bort all text som du skrivit ner i övningen?"
                                            delegate:self
                                   cancelButtonTitle:@"Forsätt"
                                   otherButtonTitles:@"Avbryt", nil];
            alert.tag=kAlertViewOne;
            [alert show];
            [alert release];
        }
        else {
            [self clearalltexts];
            raderabutton.enabled=NO;
            [list_exercise4 removeAllObjects];
            c=0;
            [list_exercise4 addObject:@"Null"];
            //isSaved = YES;
        }
    }
}


- (void)alertView:(UIAlertView *)alertView clickedButtonAtIndex:(NSInteger)buttonIndex {
    NSLog(@"ok");
    if(alertView.tag  == kAlertViewOne) {
        if (buttonIndex == 0) {
            NSLog(@"new form");
            [self clearalltexts];
            raderabutton.enabled=NO;
            [list_exercise4 removeAllObjects];
            c=0;
            [list_exercise4 addObject:@"Null"];
            //isSaved = YES;
        }else{
           //isSaved = YES;
        }
    } else if(alertView.tag == kAlertViewTwo) {
        if (buttonIndex == 0) {
            
            if (sqlite3_open([databasePath UTF8String], &exerciseDB) == SQLITE_OK) {
                
                NSString *sql = [NSString stringWithFormat: @"DELETE FROM EXERCISE4  WHERE date='%@'", [listexercise4 objectAtIndex:c] ];
                
                const char *del_stmt = [sql UTF8String];
                
                sqlite3_prepare_v2(exerciseDB, del_stmt, -1, & statement, NULL);
                if (sqlite3_step(statement) == SQLITE_ROW) {
                    
                    NSLog(@"sss");
                }
                
                sqlite3_finalize(statement);
                sqlite3_close(exerciseDB);
    
            }
            raderabutton.enabled=NO;
            [list_exercise4 removeAllObjects];
            c=0;
            [list_exercise4 addObject:@"Null"];
            [self clearalltexts];
            listofdates.hidden = YES;
            scroll.scrollEnabled = YES;
            
            UIAlertView * alert1 = [[UIAlertView alloc] initWithTitle:nil message:@"Raderat" delegate:nil cancelButtonTitle:nil otherButtonTitles:@"Ok",nil];
            [alert1 show];
            [alert1 release];
        }
        else {
            
        }
    }
}



-(void)clearalltexts {
    ex3c1.text=@"";
    ex3c2.text=@""; ex3c3.text=@""; ex3c4.text=@""; ex3c5.text=@"";
    slabel1.text=@"0%";
    slabel2.text=@"0%";
    slider.value=0.0;
    slider1.value=0.0;
}


-(IBAction)nextButton:(id)sender{
   
    scroll.scrollEnabled = NO;
    listexercise4=[[NSMutableArray alloc]init];
    [listexercise4 removeAllObjects];
    [self.view bringSubviewToFront:listofdates];
    listofdates.hidden = NO;
    [UIView beginAnimations:@"curlInView" context:nil];
    [UIView setAnimationDuration:1.0];
    [UIView commitAnimations];
      [self getlistofDates];
}





#pragma mark Loading the last saved Data
-(void)LoadSavedData {
    //SELECT * FROM EXERCISE3 WHERE date=(SELECT date FROM EXERCISE3 ORDER BY date DESC LIMIT 1)
    
    if (sqlite3_open([databasePath UTF8String], &exerciseDB) == SQLITE_OK) {
        
        NSString *sql = [NSString stringWithFormat: @"SELECT * FROM EXERCISE4 WHERE date=(SELECT date FROM EXERCISE4 ORDER BY date DESC LIMIT 1)"];
        
        const char *del_stmt = [sql UTF8String];
        
        sqlite3_prepare_v2(exerciseDB, del_stmt, -1, & statement, NULL);
        while (sqlite3_step(statement) == SQLITE_ROW) {
            
            char* c1 = (char*) sqlite3_column_text(statement,2);
            
            if (c1 != NULL){
                ex3c1.text = [NSString stringWithUTF8String:c1];
                NSLog(@"value form db :%@",ex3c1.text );
                
            }
            char* c2 = (char*) sqlite3_column_text(statement,3);
            
            if (c2 != NULL){
                ex3c2.text  = [NSString stringWithUTF8String:c2];
                NSLog(@"value form db :%@",ex3c2.text );
                
            }
            
            char* c3 = (char*) sqlite3_column_text(statement,4);
            
            if (c3!= NULL){
                ex3c3.text = [NSString stringWithUTF8String:c3];
                NSLog(@"value form db :%@",ex3c3.text );
            }
            char* c4 = (char*) sqlite3_column_text(statement,5);
            
            if (c4!= NULL){
                slabel1.text = [NSString stringWithUTF8String:c4];
                NSLog(@"value form db :%@",slabel1.text );
                int z=[slabel1.text intValue];
                float vOut = (float)z;
                slider.value=vOut;
                //slabel1.text=@"30%";
            }
            char* c5 = (char*) sqlite3_column_text(statement,6);
            
            if (c5 != NULL){
                ex3c4.text = [NSString stringWithUTF8String:c5];
                NSLog(@"value form db :%@",ex3c4.text );
                
            }
            char* c6 = (char*) sqlite3_column_text(statement,7);
            
            if (c6 != NULL){
                ex3c5.text = [NSString stringWithUTF8String:c6];
                NSLog(@"value form db :%@",ex3c5.text );
                
            }
            char* c7 = (char*) sqlite3_column_text(statement,8);
            
            if (c7 != NULL){
                NSString *str = [NSString stringWithUTF8String:c7];
                NSLog(@"value form db :%@",str);
                slabel2.text=str;
                int z=[slabel2.text intValue];
                float vOut = (float)z;
                slider1.value=vOut;
            }
        }
        
        sqlite3_finalize(statement);
        sqlite3_close(exerciseDB);
        
        
    }
    scroll.scrollEnabled = YES;
    listofdates.hidden = YES;
//    isSaved = NO;

    isSaved = YES;
    
    
    
    
    
}






-(void)getlistofDates {
    
    const char *dbpath = [databasePath UTF8String];
    
    
    if (sqlite3_open(dbpath, &exerciseDB) == SQLITE_OK)
    {
        NSString *querySQL = [NSString stringWithFormat:
                              @"SELECT date FROM EXERCISE4 ORDER BY date DESC"
                              ];
        
        const char *query_stmt = [querySQL UTF8String];
        
        if (sqlite3_prepare_v2(exerciseDB,
                               query_stmt, -1, &statement, NULL) == SQLITE_OK)
        {
            while (sqlite3_step(statement) == SQLITE_ROW) {
                
                char* date = (char*) sqlite3_column_text(statement,0);
                NSString *tmp;
                if (date != NULL){
                    tmp = [NSString stringWithUTF8String:date];
                    NSLog(@"value form db :%@",tmp);
                    [listexercise4 addObject:tmp];
                }
            }
            if (sqlite3_step(statement) != SQLITE_ROW) {
                NSLog(@"%u",listexercise4.count);
                if (listexercise4.count==0) {
                    listofdates.hidden = YES;
                    scroll.scrollEnabled=YES;
                }
            }
            sqlite3_finalize(statement);
        }
        sqlite3_close(exerciseDB);
    }
    
    [self.tableview reloadData];
}


- (NSInteger)tableView:(UITableView *)tableView numberOfRowsInSection:(NSInteger)section
{
    return [self.listexercise4 count];
}

// This will tell your UITableView what data to put in which cells in your table.
- (UITableViewCell *)tableView:(UITableView *)tableView cellForRowAtIndexPath:(NSIndexPath *)indexPath {
    static NSString *CellIdentifer = @"CellIdentifier";
    UITableViewCell *cell = [tableView dequeueReusableCellWithIdentifier:CellIdentifer];
    
    // Using a cell identifier will allow your app to reuse cells as they come and go from the screen.
    if (cell == nil) {
        cell = [[UITableViewCell alloc] initWithStyle:UITableViewCellStyleDefault reuseIdentifier:CellIdentifer];
    }
    
    // Deciding which data to put into this particular cell.
    // If it the first row, the data input will be "Data1" from the array.
    NSUInteger row = [indexPath row];
    cell.textLabel.text = [listexercise4 objectAtIndex:row];
    
    return cell;
}
- (void)tableView:(UITableView *)tableView didSelectRowAtIndexPath:(NSIndexPath *)indexPath {
	// Upon selecting an event, create an EKEventViewController to display the event.
	NSDictionary *dictionary = [self.listexercise4 objectAtIndex:indexPath.row];
    NSLog(@"%@",dictionary);
    c=indexPath.row;
    raderabutton.enabled=YES;
    SelectedDate=[NSString stringWithFormat:@"%@", dictionary];
    [list_exercise4 removeAllObjects];
    [list_exercise4 addObject:SelectedDate];
    
    NSLog(@"%@",SelectedDate);
   
   
    if (sqlite3_open([databasePath UTF8String], &exerciseDB) == SQLITE_OK) {
        
        NSString *sql = [NSString stringWithFormat: @"SELECT * FROM EXERCISE4 WHERE date='%@'", SelectedDate];
        
        const char *del_stmt = [sql UTF8String];
        
        sqlite3_prepare_v2(exerciseDB, del_stmt, -1, & statement, NULL);
        while (sqlite3_step(statement) == SQLITE_ROW) {
            
            char* c1 = (char*) sqlite3_column_text(statement,2);
            
            if (c1 != NULL){
                ex3c1.text = [NSString stringWithUTF8String:c1];
                NSLog(@"value form db :%@",ex3c1.text );
                
            }
            char* c2 = (char*) sqlite3_column_text(statement,3);
            
            if (c2 != NULL){
                ex3c2.text  = [NSString stringWithUTF8String:c2];
                NSLog(@"value form db :%@",ex3c2.text );
                
            }
            
            char* c3 = (char*) sqlite3_column_text(statement,4);
            
            if (c3!= NULL){
                ex3c3.text = [NSString stringWithUTF8String:c3];
                NSLog(@"value form db :%@",ex3c3.text );
            }
            char* c4 = (char*) sqlite3_column_text(statement,5);
            
            if (c4!= NULL){
                slabel1.text = [NSString stringWithUTF8String:c4];
                NSLog(@"value form db :%@",slabel1.text );
                int z=[slabel1.text intValue];
                float vOut = (float)z;
                slider.value=vOut;
                //slabel1.text=@"30%";
            }
            char* c5 = (char*) sqlite3_column_text(statement,6);
            
            if (c5 != NULL){
                ex3c4.text = [NSString stringWithUTF8String:c5];
                NSLog(@"value form db :%@",ex3c4.text );
                
            }
            char* c6 = (char*) sqlite3_column_text(statement,7);
            
            if (c6 != NULL){
                ex3c5.text = [NSString stringWithUTF8String:c6];
                NSLog(@"value form db :%@",ex3c5.text );
                
            }
            char* c7 = (char*) sqlite3_column_text(statement,8);
            
            if (c7 != NULL){
                NSString *str = [NSString stringWithUTF8String:c7];
                NSLog(@"value form db :%@",str);
                slabel2.text=str;
                int z=[slabel2.text intValue];
                float vOut = (float)z;
                slider1.value=vOut;
            }
        }
        
        sqlite3_finalize(statement);
        sqlite3_close(exerciseDB);
        
        
    }
    scroll.scrollEnabled = YES;
    listofdates.hidden = YES;
    isSaved = NO;
}

-(IBAction)CloseButton:(id)sender{
    scroll.scrollEnabled = YES;
     listofdates.hidden = YES;
}



- (IBAction)RaderaButton:(id)sender {
    UIAlertView *alert=[[UIAlertView alloc] initWithTitle:nil message:@"Är du säker på att du vill radera formuläret?" delegate:self cancelButtonTitle:@"Radera" otherButtonTitles:@"Avbryt", nil];
    alert.tag=kAlertViewTwo;
    [alert show];
    [alert release];
}




-(IBAction)displayDate:(id)sender{
    NSDate * selected = [picker date];
    NSDateFormatter *dateFormatter = [[NSDateFormatter alloc] init];
    [dateFormatter setDateFormat:@"yyyy-MM-dd hh:mm:ss"];
    NSString *dateString = [dateFormatter stringFromDate:selected];
    NSDate *date1 = [dateFormatter dateFromString:dateString];
    [dateFormatter setDateFormat:@"dd/MM yyyy"];
    NSString *date3  = [dateFormatter stringFromDate:date1];
    ex3c1.text=date3;
}

#pragma mark Email 

- (IBAction)skickaButtonClicked:(id)sender
{
    UIActionSheet *cameraActionSheet = [[UIActionSheet alloc] initWithTitle:@"Skicka" delegate:self cancelButtonTitle:@"Avbryt" destructiveButtonTitle:nil otherButtonTitles:@"Ladda ner", @"E-mail", nil];
    cameraActionSheet.tag = 1;
    [cameraActionSheet showInView:self.view];
}

- (UIImage *)getFormImage
{
    UIImage *tempImage = nil;
    UIGraphicsBeginImageContext(scroll.contentSize);
    {
        CGPoint savedContentOffset = scroll.contentOffset;
        CGRect savedFrame = scroll.frame;
        
        scroll.contentOffset = CGPointZero;
        scroll.frame = CGRectMake(0, 0, scroll.contentSize.width, scroll.contentSize.height);
        
        [scroll.layer renderInContext: UIGraphicsGetCurrentContext()];
        tempImage = UIGraphicsGetImageFromCurrentImageContext();
        
        scroll.contentOffset = savedContentOffset;
        scroll.frame = savedFrame;
    }
    UIGraphicsEndImageContext();
    
    return tempImage;
}

- (void)actionSheet:(UIActionSheet *)actionSheet clickedButtonAtIndex:(NSInteger)buttonIndex
{
    //    UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil);
    
	if (buttonIndex == 0)
    {
        UIImage *image = [self getFormImage];
        if (image)
        {
            UIImageWriteToSavedPhotosAlbum(image, nil, nil, nil);
            
            UIAlertView *alertView = [[UIAlertView alloc] initWithTitle:@"Alert" message:@"Image downloaded" delegate:nil cancelButtonTitle:@"Ok" otherButtonTitles:nil];
            [alertView show];
        }
    }
    else if (buttonIndex == 1)
    {
        if ([MFMailComposeViewController canSendMail])
        {
            MFMailComposeViewController *emailDialog = [[MFMailComposeViewController alloc] init];
            emailDialog.mailComposeDelegate = self;
            NSMutableString *htmlMsg = [NSMutableString string];
            [htmlMsg appendString:@"<html><body><p>"];
            [htmlMsg appendString:[NSString stringWithFormat:@"Please find the attached form on %@", SelectedDate]];
            [htmlMsg appendString:@": </p></body></html>"];
            
            NSData *jpegData = UIImageJPEGRepresentation([self getFormImage], 1);
            
            NSString *fileName = [NSString stringWithString:SelectedDate];
            fileName = [fileName stringByAppendingPathExtension:@"jpeg"];
            [emailDialog addAttachmentData:jpegData mimeType:@"image/jpeg" fileName:fileName];
            
            [emailDialog setSubject:@"Form"];
            [emailDialog setMessageBody:htmlMsg isHTML:YES];
            
            
            [self presentViewController:emailDialog animated:YES completion:nil];
        }
        else
        {
            UIAlertView *alertView = [[UIAlertView alloc] initWithTitle:@"Alert" message:@"Mail cannot be send now. Please check mail has been configured in your device and try again." delegate:nil cancelButtonTitle:@"Ok" otherButtonTitles:nil];
            [alertView show];
        }
    }
}

- (void)mailComposeController:(MFMailComposeViewController*)controller didFinishWithResult:(MFMailComposeResult)result error:(NSError*)error
{
    // Notifies users about errors associated with the interface
    switch (result)
    {
            
        case MFMailComposeResultCancelled:
            break;
        case MFMailComposeResultSaved:
            break;
        case MFMailComposeResultSent:
        {
            UIAlertView *alertView = [[UIAlertView alloc] initWithTitle:@"Alert" message:@"Mail sent successfully" delegate:nil cancelButtonTitle:@"Ok" otherButtonTitles:nil];
            [alertView show];
        }
            break;
        case MFMailComposeResultFailed:
        {
            UIAlertView *alertView = [[UIAlertView alloc] initWithTitle:@"Alert" message:@"Mail send failed" delegate:nil cancelButtonTitle:@"Ok" otherButtonTitles:nil];
            [alertView show];
        }
            break;
        default:
        {
            UIAlertView *alertView = [[UIAlertView alloc] initWithTitle:@"Alert" message:@"Mail was not sent." delegate:nil cancelButtonTitle:@"Ok" otherButtonTitles:nil];
            [alertView show];
        }
            break;
    }
    [self dismissViewControllerAnimated:YES completion:nil];
}



- (void)didReceiveMemoryWarning
{
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

@end
