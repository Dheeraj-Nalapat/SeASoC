/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE78_OS_Command_Injection__char_fromConsole_system_34.c
Label Definition File: CWE78_OS_Command_Injection.fullpath.label.xml
Template File: sources-sink-34.tmpl.c
*/
/*
 * @description
 * CWE: 78 OS Command Injection
 * BadSource: fromConsole Read input from the console
 * GoodSource: Benign input
 * Sinks: system
 *    BadSink : Execute command using system()
 * Flow Variant: 34 Data flow: use of a union containing two methods of accessing the same data (within the same function)
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

typedef union
{
    char * a;
    char * b;
} CWE78_OS_Command_Injection__char_fromConsole_system_34_union_type;

#ifndef OMITBAD

void CWE78_OS_Command_Injection__char_fromConsole_system_34_bad()
{
    char * data;
    CWE78_OS_Command_Injection__char_fromConsole_system_34_union_type my_union;
    char data_buf[100] = FULL_COMMAND;
    data = data_buf;
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
    my_union.a = data;
    {
        char * data = my_union.b;
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        system(data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    CWE78_OS_Command_Injection__char_fromConsole_system_34_union_type my_union;
    char data_buf[100] = FULL_COMMAND;
    data = data_buf;
    /* FIX: Benign input preventing command injection */
    strcat(data, "*.*");
    my_union.a = data;
    {
        char * data = my_union.b;
        /* POSSIBLE FLAW: Execute command without validating input possibly leading to command injection */
        system(data);
    }
}

void CWE78_OS_Command_Injection__char_fromConsole_system_34_good()
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
    CWE78_OS_Command_Injection__char_fromConsole_system_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE78_OS_Command_Injection__char_fromConsole_system_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
