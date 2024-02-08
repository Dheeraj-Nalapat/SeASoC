/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65a.c
Label Definition File: CWE187_Partial_Comparison.label.xml
Template File: sources-sinks-65a.tmpl.c
*/
/*
 * @description
 * CWE: 187 Partial Comparison
 * BadSource: fromConsole Read input from the console
 * GoodSource: Provide a matching password
 * Sinks: ncmp_user_pw
 *    GoodSink: Compare the 2 passwords correctly
 *    BadSink : use strncmp() to do password match, but use the length of the user password
 * Flow Variant: 65 Data/control flow: data passed as an argument from one function to a function in a different source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#define PASSWORD "Password1234"
/* PASSWORD_SZ must equal the length of PASSWORD */
#define PASSWORD_SZ strlen(PASSWORD)

#ifndef OMITBAD

/* bad function declaration */
void CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65b_bad_sink(char * data);

void CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65_bad()
{
    char * data;
    /* define a function pointer */
    void (*func_ptr) (char *) = CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65b_bad_sink;
    char data_buf[100] = "";
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
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65b_goodG2B_sink(char * data);

static void goodG2B()
{
    char * data;
    void (*func_ptr) (char *) = CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65b_goodG2B_sink;
    char data_buf[100] = "";
    data = data_buf;
    /* FIX: Use the matching string */
    data = PASSWORD;
    func_ptr(data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65b_goodB2G_sink(char * data);

static void goodB2G()
{
    char * data;
    void (*func_ptr) (char *) = CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65b_goodB2G_sink;
    char data_buf[100] = "";
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
    func_ptr(data);
}

void CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65_good()
{
    goodG2B();
    goodB2G();
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
    CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE187_Partial_Comparison__char_fromConsole_ncmp_user_pw_65_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
