/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_Environment_w32spawnvp_68b.c
Label Definition File: CWE78_OS_Command_Injection.no_path.label.xml
Template File: sources-sink-68b.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Benign input
 * Sink: w32spawnvp
 *    BadSink : execute command with spawnvp
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
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

extern char * CWE78_OS_Command_Injection__char_Environment_w32spawnvp_68_bad_data;
extern char * CWE78_OS_Command_Injection__char_Environment_w32spawnvp_68_goodG2B_data;

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

void CWE78_OS_Command_Injection__char_Environment_w32spawnvp_68b_bad_sink()
{
    char * data = CWE78_OS_Command_Injection__char_Environment_w32spawnvp_68_bad_data;
    {
        char *args[] = {COMMAND_INT_PATH, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL};
        /* spawnvp - searches for the location of the command among
         * the directories specified by the PATH environment variable */
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        _spawnvp(_P_WAIT, COMMAND_INT, args);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE78_OS_Command_Injection__char_Environment_w32spawnvp_68b_goodG2B_sink()
{
    char * data = CWE78_OS_Command_Injection__char_Environment_w32spawnvp_68_goodG2B_data;
    {
        char *args[] = {COMMAND_INT_PATH, COMMAND_ARG1, COMMAND_ARG2, COMMAND_ARG3, NULL};
        /* spawnvp - searches for the location of the command among
         * the directories specified by the PATH environment variable */
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        _spawnvp(_P_WAIT, COMMAND_INT, args);
    }
}

#endif /* OMITGOOD */
