/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67a.c
Label Definition File: CWE78_OS_Command_Injection.no_path.label.xml
Template File: sources-sink-67a.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: fromFile Read input from a file
 * GoodSource: Benign input
 * Sinks: w32spawnlp
 *    BadSink : execute command with spawnlp
 * Flow Variant: 67 Data flow: data passed in a struct from one function to another in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define COMMAND_INT_PATH "%WINDIR%\\system32\\cmd.exe"
# define COMMAND_INT "cmd.exe"
# define COMMAND_ARG1 "/c"
# define COMMAND_ARG2 "dir"
# define COMMAND_ARG3 data
#else /* NOT _WIN32 */
# define COMMAND_INT_PATH "/bin/sh"
# define COMMAND_INT "sh"
# define COMMAND_ARG1 "ls"
# define COMMAND_ARG2 data
# define COMMAND_ARG3 NULL
#endif

#ifdef _WIN32
# define FOPEN fopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

#include <process.h>

typedef struct _CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67_struct_type
{
    char * a;
} CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67_struct_type;

#ifndef OMITBAD

/* bad function declaration */
void CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67b_bad_sink(CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67_struct_type my_struct);

void CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67_bad()
{
    char * data;
    CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67_struct_type my_struct;
    char data_buf[100] = "";
    data = data_buf;
    {
        /* Read input from a file */
        size_t data_len = strlen(data);
        FILE * pFile;
        /* if there is room in data, attempt to read the input from a file */
        if(100-data_len > 1)
        {
            pFile = FOPEN("C:\\temp\\file.txt", "r");
            if (pFile != NULL)
            {
                fgets(data+data_len, (int)(100-data_len), pFile);
                fclose(pFile);
            }
        }
    }
    my_struct.a = data;
    CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67b_bad_sink(my_struct);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67b_goodG2B_sink(CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67_struct_type my_struct);

static void goodG2B()
{
    char * data;
    CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67_struct_type my_struct;
    char data_buf[100] = "";
    data = data_buf;
    /* FIX: Benign input preventing command injection */
    strcat(data, "*.*");
    my_struct.a = data;
    CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67b_goodG2B_sink(my_struct);
}

void CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67_good()
{
    goodG2B();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE78_OS_Command_Injection__char_fromFile_w32spawnlp_67_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
