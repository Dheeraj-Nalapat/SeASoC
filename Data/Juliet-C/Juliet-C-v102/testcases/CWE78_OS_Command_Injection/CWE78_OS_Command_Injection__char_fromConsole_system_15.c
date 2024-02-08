/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_fromConsole_system_15.c
Label Definition File: CWE78_OS_Command_Injection.fullpath.label.xml
Template File: sources-sink-15.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: fromConsole Read input from the console
 * GoodSource: Benign input
 * Sink: system
 *    BadSink : Execute command using system()
 * Flow Variant: 15 Control flow: switch(6)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define COMMAND_INT_PATH "%WINDIR%\\system32\\cmd.exe"
# define COMMAND_INT "cmd.exe"
# define COMMAND_ARG1 "/c"
# define COMMAND_ARG2 "dir"
# define FULL_COMMAND COMMAND_INT_PATH " " COMMAND_ARG1 " "COMMAND_ARG2 " "
#else /* NOT _WIN32 */
# define COMMAND_INT_PATH "/bin/sh"
# define COMMAND_INT "sh"
# define COMMAND_ARG1 "ls"
# define FULL_COMMAND COMMAND_INT_PATH " " COMMAND_ARG1 " "
#endif

#ifndef OMITBAD

void CWE78_OS_Command_Injection__char_fromConsole_system_15_bad()
{
    char * data;
    char data_buf[100] = FULL_COMMAND;
    data = data_buf;
    switch(6)
    {
    case 6:
    {
        /* Read input from the console */
        size_t data_len = strlen(data);
        /* if there is room in data, read into it from the console */
        if(100-data_len > 1)
        {
            fgets(data+data_len, (int)(100-data_len), stdin);
            /* The next 3 lines remove the carriage return from the string that is
             * inserted by fgets() */
            data_len = strlen(data);
            if (data_len > 0)
            {
                data[data_len-1] = '\0';
            }
        }
    }
    break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
        /* FIX: Benign input preventing command injection */
        strcat(data, "*.*");
        break;
    }
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    system(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B1() - use goodsource and badsink by changing the switch to switch(5) */
static void goodG2B1()
{
    char * data;
    char data_buf[100] = FULL_COMMAND;
    data = data_buf;
    switch(5)
    {
    case 6:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* Read input from the console */
        size_t data_len = strlen(data);
        /* if there is room in data, read into it from the console */
        if(100-data_len > 1)
        {
            fgets(data+data_len, (int)(100-data_len), stdin);
            /* The next 3 lines remove the carriage return from the string that is
             * inserted by fgets() */
            data_len = strlen(data);
            if (data_len > 0)
            {
                data[data_len-1] = '\0';
            }
        }
    }
    break;
    default:
        /* FIX: Benign input preventing command injection */
        strcat(data, "*.*");
        break;
    }
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    system(data);
}

/* goodG2B2() - use goodsource and badsink by reversing the blocks in the switch */
static void goodG2B2()
{
    char * data;
    char data_buf[100] = FULL_COMMAND;
    data = data_buf;
    switch(6)
    {
    case 6:
        /* FIX: Benign input preventing command injection */
        strcat(data, "*.*");
        break;
    default:
        /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
    {
        /* Read input from the console */
        size_t data_len = strlen(data);
        /* if there is room in data, read into it from the console */
        if(100-data_len > 1)
        {
            fgets(data+data_len, (int)(100-data_len), stdin);
            /* The next 3 lines remove the carriage return from the string that is
             * inserted by fgets() */
            data_len = strlen(data);
            if (data_len > 0)
            {
                data[data_len-1] = '\0';
            }
        }
    }
    break;
    }
    /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
    system(data);
}

void CWE78_OS_Command_Injection__char_fromConsole_system_15_good()
{
    goodG2B1();
    goodG2B2();
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
    CWE78_OS_Command_Injection__char_fromConsole_system_15_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE78_OS_Command_Injection__char_fromConsole_system_15_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
