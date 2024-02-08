/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_Environment_w32spawnlp_32.c
Label Definition File: CWE78_OS_Command_Injection.no_path.label.xml
Template File: sources-sink-32.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Benign input
 * Sink: w32spawnlp
 *    BadSink : execute command with spawnlp
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
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

#define ENV_VARIABLE "ADD"

#ifdef _WIN32
# define GETENV getenv
#else
# define GETENV getenv
#endif

#include <process.h>

#ifndef OMITBAD

void CWE78_OS_Command_Injection__char_Environment_w32spawnlp_32_bad()
{
    char * data;
    char * *data_ptr1 = &data;
    char * *data_ptr2 = &data;
    char data_buf[100] = "";
    data = data_buf;
    {
        char * data = *data_ptr1;
        {
            /* Read input from an environment variable */
            size_t data_len = strlen(data);
            char * environment = GETENV(ENV_VARIABLE);
            /* If there is data in the environment variable */
            if (environment != NULL)
            {
                strncat(data+data_len, environment, 100-data_len-1);
            }
        }
        *data_ptr1 = data;
    }
    {
        char * data = *data_ptr2;
        /* spawnlp - searches for the location of the command among
         * the directories specified by the PATH environment variable */
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        _spawnlp(_P_WAIT, COMMAND_INT, COMMAND_INT, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    char * *data_ptr1 = &data;
    char * *data_ptr2 = &data;
    char data_buf[100] = "";
    data = data_buf;
    {
        char * data = *data_ptr1;
        /* FIX: Benign input preventing command injection */
        strcat(data, "*.*");
        *data_ptr1 = data;
    }
    {
        char * data = *data_ptr2;
        /* spawnlp - searches for the location of the command among
         * the directories specified by the PATH environment variable */
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        _spawnlp(_P_WAIT, COMMAND_INT, COMMAND_INT, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL);
    }
}

void CWE78_OS_Command_Injection__char_Environment_w32spawnlp_32_good()
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
    CWE78_OS_Command_Injection__char_Environment_w32spawnlp_32_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE78_OS_Command_Injection__char_Environment_w32spawnlp_32_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
