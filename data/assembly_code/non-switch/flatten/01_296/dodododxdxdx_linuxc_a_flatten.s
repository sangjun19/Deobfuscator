	.file	"dodododxdxdx_linuxc_a_flatten.c"
	.text
	.local	task_content
	.comm	task_content,1024,32
	.globl	_TIG_IZ_OSQn_envp
	.bss
	.align 8
	.type	_TIG_IZ_OSQn_envp, @object
	.size	_TIG_IZ_OSQn_envp, 8
_TIG_IZ_OSQn_envp:
	.zero	8
	.globl	_TIG_IZ_OSQn_argv
	.align 8
	.type	_TIG_IZ_OSQn_argv, @object
	.size	_TIG_IZ_OSQn_argv, 8
_TIG_IZ_OSQn_argv:
	.zero	8
	.globl	_TIG_IZ_OSQn_argc
	.align 4
	.type	_TIG_IZ_OSQn_argc, @object
	.size	_TIG_IZ_OSQn_argc, 4
_TIG_IZ_OSQn_argc:
	.zero	4
	.local	task_file_path
	.comm	task_file_path,8,8
	.section	.rodata
.LC0:
	.string	"/tasks"
.LC1:
	.string	"\320\235\320\265\320\272\320\276\321\200\321\200\320\265\320\272\321\202\320\275\321\213\320\271 PID."
.LC2:
	.string	"\\dfh"
.LC3:
	.string	"\\mem "
.LC4:
	.string	"> "
	.align 8
.LC5:
	.string	"\320\235\320\265\320\270\320\267\320\262\320\265\321\201\321\202\320\275\320\260\321\217 \320\272\320\276\320\274\320\260\320\275\320\264\320\260: %s\n"
.LC6:
	.string	"\\bin "
.LC7:
	.string	"\\q"
.LC8:
	.string	"echo "
.LC9:
	.string	"/bin/%s"
.LC10:
	.string	"\\cron"
.LC11:
	.string	"\\l /dev/sda"
.LC12:
	.string	"-h"
.LC13:
	.string	"df"
.LC14:
	.string	"./"
.LC15:
	.string	"a"
.LC16:
	.string	"history.txt"
.LC17:
	.string	"\n"
.LC18:
	.string	"%s\n"
.LC19:
	.string	"exit"
	.text
	.globl	main
	.type	main, @function
main:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$2192, %rsp
	movl	%edi, -2164(%rbp)
	movq	%rsi, -2176(%rbp)
	movq	%rdx, -2184(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movb	$73, task_content(%rip)
	movb	$110, 1+task_content(%rip)
	movb	$105, 2+task_content(%rip)
	movb	$116, 3+task_content(%rip)
	movb	$105, 4+task_content(%rip)
	movb	$97, 5+task_content(%rip)
	movb	$108, 6+task_content(%rip)
	movb	$32, 7+task_content(%rip)
	movb	$84, 8+task_content(%rip)
	movb	$97, 9+task_content(%rip)
	movb	$115, 10+task_content(%rip)
	movb	$107, 11+task_content(%rip)
	movb	$32, 12+task_content(%rip)
	movb	$76, 13+task_content(%rip)
	movb	$105, 14+task_content(%rip)
	movb	$115, 15+task_content(%rip)
	movb	$116, 16+task_content(%rip)
	movb	$10, 17+task_content(%rip)
	movb	$0, 18+task_content(%rip)
	nop
.L2:
	leaq	.LC0(%rip), %rax
	movq	%rax, task_file_path(%rip)
	nop
.L3:
	movq	$0, _TIG_IZ_OSQn_envp(%rip)
	nop
.L4:
	movq	$0, _TIG_IZ_OSQn_argv(%rip)
	nop
.L5:
	movl	$0, _TIG_IZ_OSQn_argc(%rip)
	nop
	nop
.L6:
.L7:
#APP
# 148 "/usr/include/stdlib.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-OSQn--0
# 0 "" 2
#NO_APP
	movl	-2164(%rbp), %eax
	movl	%eax, _TIG_IZ_OSQn_argc(%rip)
	movq	-2176(%rbp), %rax
	movq	%rax, _TIG_IZ_OSQn_argv(%rip)
	movq	-2184(%rbp), %rax
	movq	%rax, _TIG_IZ_OSQn_envp(%rip)
	nop
	movq	$20, -2088(%rbp)
.L69:
	cmpq	$40, -2088(%rbp)
	ja	.L72
	movq	-2088(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L10(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L10(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L10:
	.long	.L44-.L10
	.long	.L43-.L10
	.long	.L42-.L10
	.long	.L41-.L10
	.long	.L72-.L10
	.long	.L40-.L10
	.long	.L39-.L10
	.long	.L38-.L10
	.long	.L37-.L10
	.long	.L36-.L10
	.long	.L72-.L10
	.long	.L72-.L10
	.long	.L35-.L10
	.long	.L34-.L10
	.long	.L33-.L10
	.long	.L72-.L10
	.long	.L32-.L10
	.long	.L31-.L10
	.long	.L30-.L10
	.long	.L29-.L10
	.long	.L28-.L10
	.long	.L72-.L10
	.long	.L27-.L10
	.long	.L26-.L10
	.long	.L25-.L10
	.long	.L24-.L10
	.long	.L23-.L10
	.long	.L22-.L10
	.long	.L21-.L10
	.long	.L72-.L10
	.long	.L20-.L10
	.long	.L19-.L10
	.long	.L18-.L10
	.long	.L17-.L10
	.long	.L16-.L10
	.long	.L15-.L10
	.long	.L14-.L10
	.long	.L13-.L10
	.long	.L12-.L10
	.long	.L11-.L10
	.long	.L9-.L10
	.text
.L30:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -2088(%rbp)
	jmp	.L45
.L24:
	leaq	-2064(%rbp), %rax
	addq	$5, %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -2088(%rbp)
	jmp	.L45
.L20:
	cmpl	$0, -2144(%rbp)
	jne	.L46
	movq	$19, -2088(%rbp)
	jmp	.L45
.L46:
	movq	$1, -2088(%rbp)
	jmp	.L45
.L33:
	leaq	-2064(%rbp), %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -2140(%rbp)
	movq	$40, -2088(%rbp)
	jmp	.L45
.L19:
	leaq	-2064(%rbp), %rax
	movl	$5, %edx
	leaq	.LC3(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -2136(%rbp)
	movq	$38, -2088(%rbp)
	jmp	.L45
.L35:
	call	print_partition_info
	movq	$8, -2088(%rbp)
	jmp	.L45
.L37:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	stdin(%rip), %rdx
	leaq	-2064(%rbp), %rax
	movl	$1024, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -2096(%rbp)
	movq	$13, -2088(%rbp)
	jmp	.L45
.L43:
	leaq	-2064(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -2088(%rbp)
	jmp	.L45
.L26:
	cmpl	$0, -2112(%rbp)
	jne	.L48
	movq	$39, -2088(%rbp)
	jmp	.L45
.L48:
	movq	$32, -2088(%rbp)
	jmp	.L45
.L41:
	cmpl	$0, -2128(%rbp)
	jne	.L50
	movq	$27, -2088(%rbp)
	jmp	.L45
.L50:
	movq	$22, -2088(%rbp)
	jmp	.L45
.L32:
	cmpl	$0, -2148(%rbp)
	jle	.L52
	movq	$9, -2088(%rbp)
	jmp	.L45
.L52:
	movq	$18, -2088(%rbp)
	jmp	.L45
.L25:
	leaq	-2064(%rbp), %rax
	movl	$5, %edx
	leaq	.LC6(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -2128(%rbp)
	movq	$3, -2088(%rbp)
	jmp	.L45
.L14:
	cmpl	$0, -2120(%rbp)
	jne	.L54
	movq	$25, -2088(%rbp)
	jmp	.L45
.L54:
	movq	$0, -2088(%rbp)
	jmp	.L45
.L23:
	leaq	-2064(%rbp), %rax
	leaq	.LC7(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -2112(%rbp)
	movq	$23, -2088(%rbp)
	jmp	.L45
.L36:
	movl	-2148(%rbp), %eax
	movl	%eax, %edi
	call	dump_memory
	movq	$8, -2088(%rbp)
	jmp	.L45
.L34:
	cmpq	$0, -2096(%rbp)
	je	.L56
	movq	$2, -2088(%rbp)
	jmp	.L45
.L56:
	movq	$39, -2088(%rbp)
	jmp	.L45
.L29:
	call	run_mount_script
	movq	$8, -2088(%rbp)
	jmp	.L45
.L18:
	leaq	-2064(%rbp), %rax
	movl	$5, %edx
	leaq	.LC8(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -2120(%rbp)
	movq	$36, -2088(%rbp)
	jmp	.L45
.L31:
	cmpl	$0, -2132(%rbp)
	jne	.L58
	movq	$12, -2088(%rbp)
	jmp	.L45
.L58:
	movq	$31, -2088(%rbp)
	jmp	.L45
.L9:
	cmpl	$0, -2140(%rbp)
	jne	.L60
	movq	$33, -2088(%rbp)
	jmp	.L45
.L60:
	movq	$34, -2088(%rbp)
	jmp	.L45
.L39:
	cmpl	$0, -2116(%rbp)
	jne	.L62
	movq	$39, -2088(%rbp)
	jmp	.L45
.L62:
	movq	$26, -2088(%rbp)
	jmp	.L45
.L22:
	leaq	-2064(%rbp), %rax
	addq	$5, %rax
	leaq	-1040(%rbp), %rdi
	movq	%rax, %rcx
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdx
	movl	$1024, %esi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-2064(%rbp), %rax
	addq	$5, %rax
	leaq	-1040(%rbp), %rdx
	movq	%rax, %rsi
	movq	%rdx, %rdi
	call	execute_binary
	movq	$8, -2088(%rbp)
	jmp	.L45
.L12:
	cmpl	$0, -2136(%rbp)
	jne	.L64
	movq	$37, -2088(%rbp)
	jmp	.L45
.L64:
	movq	$14, -2088(%rbp)
	jmp	.L45
.L16:
	leaq	-2064(%rbp), %rax
	leaq	.LC10(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -2144(%rbp)
	movq	$30, -2088(%rbp)
	jmp	.L45
.L27:
	leaq	-2064(%rbp), %rax
	leaq	.LC11(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -2132(%rbp)
	movq	$17, -2088(%rbp)
	jmp	.L45
.L21:
	leaq	-2064(%rbp), %rdx
	leaq	-2064(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execute_binary
	movq	$8, -2088(%rbp)
	jmp	.L45
.L40:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L70
	jmp	.L71
.L17:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	execute_binary
	movq	$8, -2088(%rbp)
	jmp	.L45
.L13:
	leaq	-2064(%rbp), %rax
	addq	$5, %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -2108(%rbp)
	movl	-2108(%rbp), %eax
	movl	%eax, -2148(%rbp)
	movq	$16, -2088(%rbp)
	jmp	.L45
.L44:
	leaq	-2064(%rbp), %rax
	movl	$2, %edx
	leaq	.LC14(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -2124(%rbp)
	movq	$7, -2088(%rbp)
	jmp	.L45
.L11:
	movq	-2104(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$5, -2088(%rbp)
	jmp	.L45
.L38:
	cmpl	$0, -2124(%rbp)
	jne	.L67
	movq	$28, -2088(%rbp)
	jmp	.L45
.L67:
	movq	$24, -2088(%rbp)
	jmp	.L45
.L15:
	leaq	.LC15(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC16(%rip), %rax
	movq	%rax, %rdi
	call	fopen64@PLT
	movq	%rax, -2080(%rbp)
	movq	-2080(%rbp), %rax
	movq	%rax, -2104(%rbp)
	leaq	handle_sighup(%rip), %rax
	movq	%rax, %rsi
	movl	$1, %edi
	call	signal@PLT
	movq	$8, -2088(%rbp)
	jmp	.L45
.L42:
	leaq	-2064(%rbp), %rax
	leaq	.LC17(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcspn@PLT
	movq	%rax, -2072(%rbp)
	leaq	-2064(%rbp), %rdx
	movq	-2072(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-2064(%rbp), %rdx
	movq	-2104(%rbp), %rax
	leaq	.LC18(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	-2104(%rbp), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	-2064(%rbp), %rax
	leaq	.LC19(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strcmp@PLT
	movl	%eax, -2116(%rbp)
	movq	$6, -2088(%rbp)
	jmp	.L45
.L28:
	movq	$35, -2088(%rbp)
	jmp	.L45
.L72:
	nop
.L45:
	jmp	.L69
.L71:
	call	__stack_chk_fail@PLT
.L70:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	main, .-main
	.section	.rodata
.LC20:
	.string	"r"
.LC21:
	.string	"/proc/partitions"
.LC22:
	.string	"sdb"
.LC23:
	.string	"sda"
	.align 8
.LC24:
	.string	"\320\230\320\275\321\204\320\276\321\200\320\274\320\260\321\206\320\270\321\217 \320\276 \321\200\320\260\320\267\320\264\320\265\320\273\320\260\321\205:"
.LC25:
	.string	"%s"
	.align 8
.LC26:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \320\276\321\202\320\272\321\200\321\213\321\202\320\270\321\217 /proc/partitions"
	.text
	.globl	print_partition_info
	.type	print_partition_info, @function
print_partition_info:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$320, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$5, -288(%rbp)
.L102:
	cmpq	$17, -288(%rbp)
	ja	.L105
	movq	-288(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L76(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L76(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L76:
	.long	.L91-.L76
	.long	.L90-.L76
	.long	.L89-.L76
	.long	.L105-.L76
	.long	.L105-.L76
	.long	.L88-.L76
	.long	.L87-.L76
	.long	.L86-.L76
	.long	.L85-.L76
	.long	.L84-.L76
	.long	.L83-.L76
	.long	.L82-.L76
	.long	.L106-.L76
	.long	.L80-.L76
	.long	.L79-.L76
	.long	.L106-.L76
	.long	.L77-.L76
	.long	.L75-.L76
	.text
.L79:
	leaq	.LC20(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC21(%rip), %rax
	movq	%rax, %rdi
	call	fopen64@PLT
	movq	%rax, -280(%rbp)
	movq	-280(%rbp), %rax
	movq	%rax, -320(%rbp)
	movq	$1, -288(%rbp)
	jmp	.L92
.L85:
	movq	-320(%rbp), %rdx
	leaq	-272(%rbp), %rax
	movl	$256, %esi
	movq	%rax, %rdi
	call	fgets@PLT
	movq	%rax, -296(%rbp)
	movq	$9, -288(%rbp)
	jmp	.L92
.L90:
	cmpq	$0, -320(%rbp)
	jne	.L94
	movq	$10, -288(%rbp)
	jmp	.L92
.L94:
	movq	$17, -288(%rbp)
	jmp	.L92
.L77:
	leaq	-272(%rbp), %rax
	leaq	.LC22(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -304(%rbp)
	movq	$2, -288(%rbp)
	jmp	.L92
.L82:
	cmpq	$0, -312(%rbp)
	je	.L96
	movq	$7, -288(%rbp)
	jmp	.L92
.L96:
	movq	$16, -288(%rbp)
	jmp	.L92
.L84:
	cmpq	$0, -296(%rbp)
	je	.L98
	movq	$13, -288(%rbp)
	jmp	.L92
.L98:
	movq	$0, -288(%rbp)
	jmp	.L92
.L80:
	leaq	-272(%rbp), %rax
	leaq	.LC23(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strstr@PLT
	movq	%rax, -312(%rbp)
	movq	$11, -288(%rbp)
	jmp	.L92
.L75:
	leaq	.LC24(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$8, -288(%rbp)
	jmp	.L92
.L87:
	leaq	-272(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -288(%rbp)
	jmp	.L92
.L88:
	movq	$14, -288(%rbp)
	jmp	.L92
.L83:
	leaq	.LC26(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$15, -288(%rbp)
	jmp	.L92
.L91:
	movq	-320(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$12, -288(%rbp)
	jmp	.L92
.L86:
	leaq	-272(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC25(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -288(%rbp)
	jmp	.L92
.L89:
	cmpq	$0, -304(%rbp)
	je	.L100
	movq	$6, -288(%rbp)
	jmp	.L92
.L100:
	movq	$8, -288(%rbp)
	jmp	.L92
.L105:
	nop
.L92:
	jmp	.L102
.L106:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L104
	call	__stack_chk_fail@PLT
.L104:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	print_partition_info, .-print_partition_info
	.section	.rodata
	.align 8
.LC27:
	.string	"\320\241\320\272\321\200\320\270\320\277\321\202 \320\267\320\260\320\262\320\265\321\200\321\210\321\221\320\275 \321\201 \320\272\320\276\320\264\320\276\320\274 %d\n"
	.align 8
.LC28:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \320\262\321\213\320\277\320\276\320\273\320\275\320\265\320\275\320\270\321\217 \321\201\320\272\321\200\320\270\320\277\321\202\320\260"
	.align 8
.LC29:
	.string	"/home/rodion/Linux_homework-main/Bogdanov_Rodion_24/mount_vfs.sh"
	.align 8
.LC30:
	.string	"\320\241\320\272\321\200\320\270\320\277\321\202 \320\267\320\260\320\262\320\265\321\200\321\210\320\270\320\273\321\201\321\217 \320\275\320\265\320\275\320\276\321\200\320\274\320\260\320\273\321\214\320\275\320\276"
	.align 8
.LC31:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \321\201\320\276\320\267\320\264\320\260\320\275\320\270\321\217 \320\277\321\200\320\276\321\206\320\265\321\201\321\201\320\260 \320\264\320\273\321\217 \321\201\320\272\321\200\320\270\320\277\321\202\320\260"
	.align 8
.LC32:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260: \321\201\320\272\321\200\320\270\320\277\321\202 \320\275\320\265\320\264\320\276\321\201\321\202\321\203\320\277\320\265\320\275 \320\264\320\273\321\217 \320\262\321\213\320\277\320\276\320\273\320\275\320\265\320\275\320\270\321\217"
	.text
	.globl	run_mount_script
	.type	run_mount_script, @function
run_mount_script:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$6, -16(%rbp)
.L135:
	cmpq	$16, -16(%rbp)
	ja	.L138
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L110(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L110(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L110:
	.long	.L124-.L110
	.long	.L123-.L110
	.long	.L122-.L110
	.long	.L121-.L110
	.long	.L120-.L110
	.long	.L138-.L110
	.long	.L119-.L110
	.long	.L118-.L110
	.long	.L117-.L110
	.long	.L116-.L110
	.long	.L115-.L110
	.long	.L114-.L110
	.long	.L139-.L110
	.long	.L112-.L110
	.long	.L138-.L110
	.long	.L111-.L110
	.long	.L139-.L110
	.text
.L120:
	call	fork@PLT
	movl	%eax, -28(%rbp)
	movl	-28(%rbp), %eax
	movl	%eax, -32(%rbp)
	movq	$11, -16(%rbp)
	jmp	.L125
.L111:
	movl	-40(%rbp), %eax
	sarl	$8, %eax
	movzbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC27(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$12, -16(%rbp)
	jmp	.L125
.L117:
	movq	-24(%rbp), %rcx
	movq	-24(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	execl@PLT
	leaq	.LC28(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L123:
	leaq	-40(%rbp), %rcx
	movl	-32(%rbp), %eax
	movl	$0, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	waitpid@PLT
	movq	$7, -16(%rbp)
	jmp	.L125
.L121:
	leaq	.LC29(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	access@PLT
	movl	%eax, -36(%rbp)
	movq	$2, -16(%rbp)
	jmp	.L125
.L114:
	cmpl	$0, -32(%rbp)
	jne	.L127
	movq	$8, -16(%rbp)
	jmp	.L125
.L127:
	movq	$9, -16(%rbp)
	jmp	.L125
.L116:
	cmpl	$0, -32(%rbp)
	jle	.L129
	movq	$1, -16(%rbp)
	jmp	.L125
.L129:
	movq	$10, -16(%rbp)
	jmp	.L125
.L112:
	leaq	.LC30(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -16(%rbp)
	jmp	.L125
.L119:
	movq	$3, -16(%rbp)
	jmp	.L125
.L115:
	leaq	.LC31(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$12, -16(%rbp)
	jmp	.L125
.L124:
	leaq	.LC32(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$16, -16(%rbp)
	jmp	.L125
.L118:
	movl	-40(%rbp), %eax
	andl	$127, %eax
	testl	%eax, %eax
	jne	.L131
	movq	$15, -16(%rbp)
	jmp	.L125
.L131:
	movq	$13, -16(%rbp)
	jmp	.L125
.L122:
	cmpl	$-1, -36(%rbp)
	jne	.L133
	movq	$0, -16(%rbp)
	jmp	.L125
.L133:
	movq	$4, -16(%rbp)
	jmp	.L125
.L138:
	nop
.L125:
	jmp	.L135
.L139:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L137
	call	__stack_chk_fail@PLT
.L137:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	run_mount_script, .-run_mount_script
	.section	.rodata
.LC33:
	.string	"Configuration reloaded"
	.text
	.globl	handle_sighup
	.type	handle_sighup, @function
handle_sighup:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$0, -8(%rbp)
.L145:
	cmpq	$0, -8(%rbp)
	je	.L141
	cmpq	$1, -8(%rbp)
	jne	.L147
	jmp	.L146
.L141:
	leaq	.LC33(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L144
.L147:
	nop
.L144:
	jmp	.L145
.L146:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	handle_sighup, .-handle_sighup
	.section	.rodata
.LC34:
	.string	" "
	.align 8
.LC35:
	.string	"\320\237\321\200\320\276\320\263\321\200\320\260\320\274\320\274\320\260 \320\267\320\260\320\262\320\265\321\200\321\210\320\270\320\273\320\260\321\201\321\214 \320\275\320\265\320\275\320\276\321\200\320\274\320\260\320\273\321\214\320\275\320\276"
	.align 8
.LC36:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \320\262\321\213\320\277\320\276\320\273\320\275\320\265\320\275\320\270\321\217 \320\272\320\276\320\274\320\260\320\275\320\264\321\213"
	.align 8
.LC37:
	.string	"\320\237\321\200\320\276\320\263\321\200\320\260\320\274\320\274\320\260 \320\267\320\260\320\262\320\265\321\200\321\210\320\265\320\275\320\260 \321\201 \320\272\320\276\320\264\320\276\320\274 %d\n"
	.align 8
.LC38:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \321\201\320\276\320\267\320\264\320\260\320\275\320\270\321\217 \320\277\321\200\320\276\321\206\320\265\321\201\321\201\320\260"
	.align 8
.LC39:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260: \320\275\320\265\320\264\320\276\321\201\321\202\320\260\321\202\320\276\321\207\320\275\320\276 \320\277\321\200\320\260\320\262 \320\264\320\273\321\217 \320\262\321\213\320\277\320\276\320\273\320\275\320\265\320\275\320\270\321\217 \321\204\320\260\320\271\320\273\320\260 \320\270\320\273\320\270 \321\204\320\260\320\271\320\273 \320\275\320\265 \320\275\320\260\320\271\320\264\320\265\320\275"
	.text
	.globl	execute_binary
	.type	execute_binary, @function
execute_binary:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$592, %rsp
	movq	%rdi, -584(%rbp)
	movq	%rsi, -592(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$24, -544(%rbp)
.L184:
	cmpq	$25, -544(%rbp)
	ja	.L187
	movq	-544(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L151(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L151(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L151:
	.long	.L187-.L151
	.long	.L169-.L151
	.long	.L168-.L151
	.long	.L188-.L151
	.long	.L166-.L151
	.long	.L165-.L151
	.long	.L164-.L151
	.long	.L163-.L151
	.long	.L162-.L151
	.long	.L187-.L151
	.long	.L188-.L151
	.long	.L160-.L151
	.long	.L159-.L151
	.long	.L187-.L151
	.long	.L187-.L151
	.long	.L158-.L151
	.long	.L187-.L151
	.long	.L157-.L151
	.long	.L156-.L151
	.long	.L155-.L151
	.long	.L154-.L151
	.long	.L153-.L151
	.long	.L187-.L151
	.long	.L187-.L151
	.long	.L152-.L151
	.long	.L150-.L151
	.text
.L156:
	movl	$0, -572(%rbp)
	movq	-592(%rbp), %rax
	leaq	.LC34(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -536(%rbp)
	movq	-536(%rbp), %rax
	movq	%rax, -552(%rbp)
	movq	$11, -544(%rbp)
	jmp	.L170
.L150:
	leaq	.LC35(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$10, -544(%rbp)
	jmp	.L170
.L166:
	leaq	-528(%rbp), %rdx
	movq	-584(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	execvp@PLT
	leaq	.LC36(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L158:
	cmpl	$62, -572(%rbp)
	jg	.L171
	movq	$12, -544(%rbp)
	jmp	.L170
.L171:
	movq	$5, -544(%rbp)
	jmp	.L170
.L159:
	movl	-572(%rbp), %eax
	movl	%eax, -556(%rbp)
	addl	$1, -572(%rbp)
	movl	-556(%rbp), %eax
	cltq
	movq	-552(%rbp), %rdx
	movq	%rdx, -528(%rbp,%rax,8)
	leaq	.LC34(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -552(%rbp)
	movq	$11, -544(%rbp)
	jmp	.L170
.L162:
	cmpl	$0, -564(%rbp)
	jle	.L173
	movq	$19, -544(%rbp)
	jmp	.L170
.L173:
	movq	$2, -544(%rbp)
	jmp	.L170
.L169:
	movl	-576(%rbp), %eax
	sarl	$8, %eax
	movzbl	%al, %eax
	movl	%eax, %esi
	leaq	.LC37(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$10, -544(%rbp)
	jmp	.L170
.L152:
	movq	$18, -544(%rbp)
	jmp	.L170
.L153:
	movl	-576(%rbp), %eax
	andl	$127, %eax
	testl	%eax, %eax
	jne	.L176
	movq	$1, -544(%rbp)
	jmp	.L170
.L176:
	movq	$25, -544(%rbp)
	jmp	.L170
.L160:
	cmpq	$0, -552(%rbp)
	je	.L178
	movq	$15, -544(%rbp)
	jmp	.L170
.L178:
	movq	$5, -544(%rbp)
	jmp	.L170
.L155:
	leaq	-576(%rbp), %rcx
	movl	-564(%rbp), %eax
	movl	$0, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	waitpid@PLT
	movq	$21, -544(%rbp)
	jmp	.L170
.L157:
	call	fork@PLT
	movl	%eax, -560(%rbp)
	movl	-560(%rbp), %eax
	movl	%eax, -564(%rbp)
	movq	$7, -544(%rbp)
	jmp	.L170
.L164:
	cmpl	$-1, -568(%rbp)
	jne	.L180
	movq	$20, -544(%rbp)
	jmp	.L170
.L180:
	movq	$17, -544(%rbp)
	jmp	.L170
.L165:
	movl	-572(%rbp), %eax
	cltq
	movq	$0, -528(%rbp,%rax,8)
	movq	-584(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	access@PLT
	movl	%eax, -568(%rbp)
	movq	$6, -544(%rbp)
	jmp	.L170
.L163:
	cmpl	$0, -564(%rbp)
	jne	.L182
	movq	$4, -544(%rbp)
	jmp	.L170
.L182:
	movq	$8, -544(%rbp)
	jmp	.L170
.L168:
	leaq	.LC38(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$10, -544(%rbp)
	jmp	.L170
.L154:
	leaq	.LC39(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$3, -544(%rbp)
	jmp	.L170
.L187:
	nop
.L170:
	jmp	.L184
.L188:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L186
	call	__stack_chk_fail@PLT
.L186:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	execute_binary, .-execute_binary
	.section	.rodata
	.align 8
.LC40:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \321\207\321\202\320\265\320\275\320\270\321\217 \321\201\321\201\321\213\320\273\320\272\320\270 \320\275\320\260 \321\201\320\265\320\263\320\274\320\265\320\275\321\202 \320\277\320\260\320\274\321\217\321\202\320\270"
.LC41:
	.string	"%s/%s"
	.align 8
.LC42:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260: \320\264\320\273\321\217 \320\264\320\260\320\274\320\277\320\260 \320\277\320\260\320\274\321\217\321\202\320\270 \320\277\321\200\320\276\321\206\320\265\321\201\321\201\320\260 \320\275\321\203\320\266\320\275\321\213 \320\277\321\200\320\260\320\262\320\260 root."
	.align 8
.LC43:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \320\276\321\202\320\272\321\200\321\213\321\202\320\270\321\217 \320\264\320\270\321\200\320\265\320\272\321\202\320\276\321\200\320\270\320\270 map_files"
.LC44:
	.string	"/tmp/memory_dumps_%d"
.LC45:
	.string	"%s/%s.bin"
	.align 8
.LC46:
	.string	"\320\241\320\265\320\263\320\274\320\265\320\275\321\202 \320\277\320\260\320\274\321\217\321\202\320\270 %s \320\264\320\260\320\274\320\277\320\270\321\200\320\276\320\262\320\260\320\275.\n"
.LC47:
	.string	"/proc/%d/map_files"
	.text
	.globl	dump_memory
	.type	dump_memory, @function
dump_memory:
.LFB11:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1824, %rsp
	movl	%edi, -1812(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$4, -1768(%rbp)
.L222:
	cmpq	$23, -1768(%rbp)
	ja	.L225
	movq	-1768(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L192(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L192(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L192:
	.long	.L226-.L192
	.long	.L225-.L192
	.long	.L226-.L192
	.long	.L225-.L192
	.long	.L207-.L192
	.long	.L226-.L192
	.long	.L205-.L192
	.long	.L204-.L192
	.long	.L203-.L192
	.long	.L202-.L192
	.long	.L201-.L192
	.long	.L200-.L192
	.long	.L225-.L192
	.long	.L199-.L192
	.long	.L225-.L192
	.long	.L198-.L192
	.long	.L225-.L192
	.long	.L197-.L192
	.long	.L196-.L192
	.long	.L195-.L192
	.long	.L194-.L192
	.long	.L193-.L192
	.long	.L225-.L192
	.long	.L191-.L192
	.text
.L196:
	leaq	.LC40(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$23, -1768(%rbp)
	jmp	.L210
.L207:
	call	geteuid@PLT
	movl	%eax, -1796(%rbp)
	movq	$13, -1768(%rbp)
	jmp	.L210
.L198:
	cmpq	$0, -1784(%rbp)
	je	.L211
	movq	$6, -1768(%rbp)
	jmp	.L210
.L211:
	movq	$20, -1768(%rbp)
	jmp	.L210
.L203:
	movq	-1784(%rbp), %rax
	leaq	19(%rax), %rcx
	leaq	-1744(%rbp), %rdx
	leaq	-1552(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC41(%rip), %rdx
	movl	$512, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1040(%rbp), %rcx
	leaq	-1552(%rbp), %rax
	movl	$511, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	readlink@PLT
	movq	%rax, -1760(%rbp)
	movq	-1760(%rbp), %rax
	movq	%rax, -1776(%rbp)
	movq	$17, -1768(%rbp)
	jmp	.L210
.L191:
	movq	-1792(%rbp), %rax
	movq	%rax, %rdi
	call	readdir64@PLT
	movq	%rax, -1784(%rbp)
	movq	$15, -1768(%rbp)
	jmp	.L210
.L193:
	cmpq	$0, -1792(%rbp)
	jne	.L213
	movq	$9, -1768(%rbp)
	jmp	.L210
.L213:
	movq	$19, -1768(%rbp)
	jmp	.L210
.L200:
	leaq	.LC42(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$5, -1768(%rbp)
	jmp	.L210
.L202:
	leaq	.LC43(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$0, -1768(%rbp)
	jmp	.L210
.L199:
	cmpl	$0, -1796(%rbp)
	je	.L215
	movq	$11, -1768(%rbp)
	jmp	.L210
.L215:
	movq	$7, -1768(%rbp)
	jmp	.L210
.L195:
	movl	-1812(%rbp), %edx
	leaq	-1680(%rbp), %rax
	movl	%edx, %ecx
	leaq	.LC44(%rip), %rdx
	movl	$128, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1680(%rbp), %rax
	movl	$493, %esi
	movq	%rax, %rdi
	call	mkdir@PLT
	movq	$23, -1768(%rbp)
	jmp	.L210
.L197:
	cmpq	$-1, -1776(%rbp)
	jne	.L217
	movq	$18, -1768(%rbp)
	jmp	.L210
.L217:
	movq	$10, -1768(%rbp)
	jmp	.L210
.L205:
	movq	-1784(%rbp), %rax
	movzbl	18(%rax), %eax
	cmpb	$10, %al
	jne	.L219
	movq	$8, -1768(%rbp)
	jmp	.L210
.L219:
	movq	$23, -1768(%rbp)
	jmp	.L210
.L201:
	leaq	-1040(%rbp), %rdx
	movq	-1776(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	movq	-1784(%rbp), %rax
	leaq	19(%rax), %rcx
	leaq	-1680(%rbp), %rdx
	leaq	-528(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC45(%rip), %rdx
	movl	$512, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-528(%rbp), %rdx
	leaq	-1040(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	dump_memory_segment
	movq	-1784(%rbp), %rax
	addq	$19, %rax
	movq	%rax, %rsi
	leaq	.LC46(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$23, -1768(%rbp)
	jmp	.L210
.L204:
	movl	-1812(%rbp), %edx
	leaq	-1744(%rbp), %rax
	movl	%edx, %ecx
	leaq	.LC47(%rip), %rdx
	movl	$64, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1744(%rbp), %rax
	movq	%rax, %rdi
	call	opendir@PLT
	movq	%rax, -1752(%rbp)
	movq	-1752(%rbp), %rax
	movq	%rax, -1792(%rbp)
	movq	$21, -1768(%rbp)
	jmp	.L210
.L194:
	movq	-1792(%rbp), %rax
	movq	%rax, %rdi
	call	closedir@PLT
	movq	$2, -1768(%rbp)
	jmp	.L210
.L225:
	nop
.L210:
	jmp	.L222
.L226:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L224
	call	__stack_chk_fail@PLT
.L224:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE11:
	.size	dump_memory, .-dump_memory
	.section	.rodata
	.align 8
.LC48:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \321\201\320\276\320\267\320\264\320\260\320\275\320\270\321\217 \320\264\320\260\320\274\320\277-\321\204\320\260\320\271\320\273\320\260 \320\264\320\273\321\217 \321\201\320\265\320\263\320\274\320\265\320\275\321\202\320\260"
	.align 8
.LC49:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \321\207\321\202\320\265\320\275\320\270\321\217 \320\277\320\260\320\274\321\217\321\202\320\270 \321\201\320\265\320\263\320\274\320\265\320\275\321\202\320\260"
	.align 8
.LC50:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \320\267\320\260\320\277\320\270\321\201\320\270 \320\262 \320\264\320\260\320\274\320\277-\321\204\320\260\320\271\320\273 \321\201\320\265\320\263\320\274\320\265\320\275\321\202\320\260"
	.align 8
.LC51:
	.string	"\320\236\321\210\320\270\320\261\320\272\320\260 \320\276\321\202\320\272\321\200\321\213\321\202\320\270\321\217 \321\204\320\260\320\271\320\273\320\260 \321\201\320\265\320\263\320\274\320\265\320\275\321\202\320\260 \320\277\320\260\320\274\321\217\321\202\320\270"
	.text
	.globl	dump_memory_segment
	.type	dump_memory_segment, @function
dump_memory_segment:
.LFB12:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$80, %rsp
	movq	%rdi, -4168(%rbp)
	movq	%rsi, -4176(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -4120(%rbp)
.L260:
	cmpq	$23, -4120(%rbp)
	ja	.L263
	movq	-4120(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L230(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L230(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L230:
	.long	.L264-.L230
	.long	.L263-.L230
	.long	.L246-.L230
	.long	.L245-.L230
	.long	.L244-.L230
	.long	.L243-.L230
	.long	.L264-.L230
	.long	.L241-.L230
	.long	.L240-.L230
	.long	.L239-.L230
	.long	.L263-.L230
	.long	.L238-.L230
	.long	.L237-.L230
	.long	.L236-.L230
	.long	.L263-.L230
	.long	.L235-.L230
	.long	.L234-.L230
	.long	.L263-.L230
	.long	.L233-.L230
	.long	.L264-.L230
	.long	.L231-.L230
	.long	.L263-.L230
	.long	.L263-.L230
	.long	.L229-.L230
	.text
.L233:
	cmpq	$-1, -4136(%rbp)
	jne	.L248
	movq	$5, -4120(%rbp)
	jmp	.L250
.L248:
	movq	$23, -4120(%rbp)
	jmp	.L250
.L244:
	leaq	.LC48(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-4152(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$0, -4120(%rbp)
	jmp	.L250
.L235:
	cmpl	$-1, -4148(%rbp)
	jne	.L251
	movq	$4, -4120(%rbp)
	jmp	.L250
.L251:
	movq	$13, -4120(%rbp)
	jmp	.L250
.L237:
	movq	-4136(%rbp), %rdx
	leaq	-4112(%rbp), %rcx
	movl	-4148(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	write@PLT
	movq	%rax, -4128(%rbp)
	movq	$11, -4120(%rbp)
	jmp	.L250
.L240:
	movq	-4176(%rbp), %rax
	movl	$420, %edx
	movl	$577, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open64@PLT
	movl	%eax, -4140(%rbp)
	movl	-4140(%rbp), %eax
	movl	%eax, -4148(%rbp)
	movq	$15, -4120(%rbp)
	jmp	.L250
.L229:
	movl	-4152(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-4148(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movq	$6, -4120(%rbp)
	jmp	.L250
.L245:
	cmpl	$-1, -4152(%rbp)
	jne	.L253
	movq	$20, -4120(%rbp)
	jmp	.L250
.L253:
	movq	$8, -4120(%rbp)
	jmp	.L250
.L234:
	cmpq	$0, -4136(%rbp)
	jle	.L255
	movq	$12, -4120(%rbp)
	jmp	.L250
.L255:
	movq	$18, -4120(%rbp)
	jmp	.L250
.L238:
	movq	-4128(%rbp), %rax
	cmpq	-4136(%rbp), %rax
	je	.L257
	movq	$7, -4120(%rbp)
	jmp	.L250
.L257:
	movq	$13, -4120(%rbp)
	jmp	.L250
.L239:
	movq	-4168(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	open64@PLT
	movl	%eax, -4144(%rbp)
	movl	-4144(%rbp), %eax
	movl	%eax, -4152(%rbp)
	movq	$3, -4120(%rbp)
	jmp	.L250
.L236:
	leaq	-4112(%rbp), %rcx
	movl	-4152(%rbp), %eax
	movl	$4096, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	read@PLT
	movq	%rax, -4136(%rbp)
	movq	$16, -4120(%rbp)
	jmp	.L250
.L243:
	leaq	.LC49(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$23, -4120(%rbp)
	jmp	.L250
.L241:
	leaq	.LC50(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$18, -4120(%rbp)
	jmp	.L250
.L246:
	movq	$9, -4120(%rbp)
	jmp	.L250
.L231:
	leaq	.LC51(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movq	$19, -4120(%rbp)
	jmp	.L250
.L263:
	nop
.L250:
	jmp	.L260
.L264:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L262
	call	__stack_chk_fail@PLT
.L262:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE12:
	.size	dump_memory_segment, .-dump_memory_segment
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
