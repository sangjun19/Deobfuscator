	.file	"mkalin_courses_signalsJumps_flatten.c"
	.text
	.local	env
	.comm	env,200,32
	.globl	_TIG_IZ_3o2x_argv
	.bss
	.align 8
	.type	_TIG_IZ_3o2x_argv, @object
	.size	_TIG_IZ_3o2x_argv, 8
_TIG_IZ_3o2x_argv:
	.zero	8
	.globl	_TIG_IZ_3o2x_envp
	.align 8
	.type	_TIG_IZ_3o2x_envp, @object
	.size	_TIG_IZ_3o2x_envp, 8
_TIG_IZ_3o2x_envp:
	.zero	8
	.globl	_TIG_IZ_3o2x_argc
	.align 4
	.type	_TIG_IZ_3o2x_argc, @object
	.size	_TIG_IZ_3o2x_argc, 4
_TIG_IZ_3o2x_argc:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"\t Control to jumper: signal status == %d.\n"
	.text
	.globl	jumper
	.type	jumper, @function
jumper:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$2, -8(%rbp)
.L6:
	cmpq	$0, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	je	.L3
	jmp	.L5
.L2:
	movl	-20(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, %esi
	leaq	env(%rip), %rax
	movq	%rax, %rdi
	call	longjmp@PLT
.L3:
	movq	$0, -8(%rbp)
	nop
.L5:
	jmp	.L6
	.cfi_endproc
.LFE1:
	.size	jumper, .-jumper
	.section	.rodata
.LC1:
	.string	"Right after divisionInt()...."
	.text
	.globl	guard
	.type	guard, @function
guard:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L14:
	cmpq	$2, -8(%rbp)
	je	.L8
	cmpq	$2, -8(%rbp)
	ja	.L17
	cmpq	$0, -8(%rbp)
	je	.L15
	cmpq	$1, -8(%rbp)
	jne	.L17
	movl	$0, %eax
	jmp	.L16
.L15:
	leaq	env(%rip), %rax
	movq	%rax, %rdi
	call	_setjmp@PLT
	endbr64
	call	divisionInt
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$1, -8(%rbp)
	jmp	.L13
.L8:
	movq	$0, -8(%rbp)
	jmp	.L13
.L17:
	nop
.L13:
	jmp	.L14
.L16:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	guard, .-guard
	.globl	trace_and_die
	.type	trace_and_die, @function
trace_and_die:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$0, -8(%rbp)
.L23:
	cmpq	$0, -8(%rbp)
	je	.L19
	cmpq	$2, -8(%rbp)
	je	.L20
	jmp	.L22
.L19:
	movq	$2, -8(%rbp)
	jmp	.L22
.L20:
	movq	stderr(%rip), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fputs@PLT
	movl	$1, %edi
	call	exit@PLT
.L22:
	jmp	.L23
	.cfi_endproc
.LFE3:
	.size	trace_and_die, .-trace_and_die
	.section	.rodata
.LC2:
	.string	"SIGFPE is defined as %i\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -20(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	$0, env(%rip)
	movq	$0, 8+env(%rip)
	movq	$0, 16+env(%rip)
	movq	$0, 24+env(%rip)
	movq	$0, 32+env(%rip)
	movq	$0, 40+env(%rip)
	movq	$0, 48+env(%rip)
	movq	$0, 56+env(%rip)
	movl	$0, 64+env(%rip)
	movq	$0, 72+env(%rip)
	movq	$0, 80+env(%rip)
	movq	$0, 88+env(%rip)
	movq	$0, 96+env(%rip)
	movq	$0, 104+env(%rip)
	movq	$0, 112+env(%rip)
	movq	$0, 120+env(%rip)
	movq	$0, 128+env(%rip)
	movq	$0, 136+env(%rip)
	movq	$0, 144+env(%rip)
	movq	$0, 152+env(%rip)
	movq	$0, 160+env(%rip)
	movq	$0, 168+env(%rip)
	movq	$0, 176+env(%rip)
	movq	$0, 184+env(%rip)
	movq	$0, 192+env(%rip)
	nop
.L25:
	movq	$0, _TIG_IZ_3o2x_envp(%rip)
	nop
.L26:
	movq	$0, _TIG_IZ_3o2x_argv(%rip)
	nop
.L27:
	movl	$0, _TIG_IZ_3o2x_argc(%rip)
	nop
	nop
.L28:
.L29:
#APP
# 119 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-3o2x--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_3o2x_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_3o2x_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_3o2x_envp(%rip)
	nop
	movq	$3, -8(%rbp)
.L41:
	cmpq	$6, -8(%rbp)
	ja	.L43
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L32(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L32(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L32:
	.long	.L36-.L32
	.long	.L35-.L32
	.long	.L43-.L32
	.long	.L34-.L32
	.long	.L43-.L32
	.long	.L33-.L32
	.long	.L31-.L32
	.text
.L35:
	call	guard
	movl	%eax, -12(%rbp)
	movq	$6, -8(%rbp)
	jmp	.L37
.L34:
	movq	$0, -8(%rbp)
	jmp	.L37
.L31:
	cmpl	$0, -12(%rbp)
	jne	.L38
	movq	$1, -8(%rbp)
	jmp	.L37
.L38:
	movq	$5, -8(%rbp)
	jmp	.L37
.L33:
	movl	$0, %eax
	jmp	.L42
.L36:
	movl	$8, %esi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	jumper(%rip), %rax
	movq	%rax, %rsi
	movl	$8, %edi
	call	signal@PLT
	movq	$1, -8(%rbp)
	jmp	.L37
.L43:
	nop
.L37:
	jmp	.L41
.L42:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	main, .-main
	.section	.rodata
.LC3:
	.string	"Two integers: "
.LC4:
	.string	"%i %i"
.LC5:
	.string	"%i / %i == %i\n"
	.text
	.globl	divisionInt
	.type	divisionInt, @function
divisionInt:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$2, -16(%rbp)
.L50:
	cmpq	$2, -16(%rbp)
	je	.L45
	cmpq	$2, -16(%rbp)
	ja	.L53
	cmpq	$0, -16(%rbp)
	je	.L54
	cmpq	$1, -16(%rbp)
	jne	.L53
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-20(%rbp), %rdx
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movl	-24(%rbp), %eax
	movl	-20(%rbp), %esi
	cltd
	idivl	%esi
	movl	%eax, %ecx
	movl	-20(%rbp), %edx
	movl	-24(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$0, -16(%rbp)
	jmp	.L48
.L45:
	movq	$1, -16(%rbp)
	jmp	.L48
.L53:
	nop
.L48:
	jmp	.L50
.L54:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L52
	call	__stack_chk_fail@PLT
.L52:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	divisionInt, .-divisionInt
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
