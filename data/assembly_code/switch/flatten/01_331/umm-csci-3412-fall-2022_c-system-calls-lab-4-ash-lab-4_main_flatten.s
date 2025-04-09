	.file	"umm-csci-3412-fall-2022_c-system-calls-lab-4-ash-lab-4_main_flatten.c"
	.text
	.globl	_TIG_IZ_D8V6_argc
	.bss
	.align 4
	.type	_TIG_IZ_D8V6_argc, @object
	.size	_TIG_IZ_D8V6_argc, 4
_TIG_IZ_D8V6_argc:
	.zero	4
	.globl	_TIG_IZ_D8V6_argv
	.align 8
	.type	_TIG_IZ_D8V6_argv, @object
	.size	_TIG_IZ_D8V6_argv, 8
_TIG_IZ_D8V6_argv:
	.zero	8
	.globl	_TIG_IZ_D8V6_envp
	.align 8
	.type	_TIG_IZ_D8V6_envp, @object
	.size	_TIG_IZ_D8V6_envp, 8
_TIG_IZ_D8V6_envp:
	.zero	8
	.text
	.globl	disemvowel
	.type	disemvowel, @function
disemvowel:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$8, -24(%rbp)
.L16:
	cmpq	$9, -24(%rbp)
	ja	.L19
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L9-.L4
	.long	.L8-.L4
	.long	.L20-.L4
	.long	.L6-.L4
	.long	.L19-.L4
	.long	.L19-.L4
	.long	.L19-.L4
	.long	.L19-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	movq	$0, -24(%rbp)
	jmp	.L10
.L8:
	movq	-48(%rbp), %rdx
	leaq	-29(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fwrite@PLT
	movq	$9, -24(%rbp)
	jmp	.L10
.L6:
	movzbl	-29(%rbp), %eax
	movsbl	%al, %eax
	subl	$65, %eax
	cmpl	$52, %eax
	seta	%dl
	testb	%dl, %dl
	jne	.L11
	movabsq	$4575140898685201, %rdx
	movl	%eax, %ecx
	shrq	%cl, %rdx
	movq	%rdx, %rax
	andl	$1, %eax
	testq	%rax, %rax
	setne	%al
	testb	%al, %al
	je	.L11
	movq	$9, -24(%rbp)
	jmp	.L12
.L11:
	movq	$1, -24(%rbp)
	nop
.L12:
	jmp	.L10
.L3:
	cmpl	$0, -28(%rbp)
	je	.L13
	movq	$0, -24(%rbp)
	jmp	.L10
.L13:
	movq	$2, -24(%rbp)
	jmp	.L10
.L9:
	movq	-40(%rbp), %rdx
	leaq	-29(%rbp), %rax
	movq	%rdx, %rcx
	movl	$1, %edx
	movl	$1, %esi
	movq	%rax, %rdi
	call	fread@PLT
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movl	%eax, -28(%rbp)
	movq	$3, -24(%rbp)
	jmp	.L10
.L19:
	nop
.L10:
	jmp	.L16
.L20:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L18
	call	__stack_chk_fail@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	disemvowel, .-disemvowel
	.section	.rodata
.LC0:
	.string	"unable to open file '%s'\n"
.LC1:
	.string	"r"
.LC2:
	.string	"w"
	.text
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	$0, _TIG_IZ_D8V6_envp(%rip)
	nop
.L22:
	movq	$0, _TIG_IZ_D8V6_argv(%rip)
	nop
.L23:
	movl	$0, _TIG_IZ_D8V6_argc(%rip)
	nop
	nop
.L24:
.L25:
#APP
# 125 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-D8V6--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_D8V6_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_D8V6_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_D8V6_envp(%rip)
	nop
	movq	$11, -8(%rbp)
.L52:
	cmpq	$14, -8(%rbp)
	ja	.L53
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L28(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L28(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L28:
	.long	.L41-.L28
	.long	.L40-.L28
	.long	.L39-.L28
	.long	.L38-.L28
	.long	.L37-.L28
	.long	.L53-.L28
	.long	.L36-.L28
	.long	.L35-.L28
	.long	.L34-.L28
	.long	.L33-.L28
	.long	.L32-.L28
	.long	.L31-.L28
	.long	.L30-.L28
	.long	.L29-.L28
	.long	.L27-.L28
	.text
.L37:
	cmpl	$1, -36(%rbp)
	jle	.L42
	movq	$6, -8(%rbp)
	jmp	.L44
.L42:
	movq	$2, -8(%rbp)
	jmp	.L44
.L27:
	movq	stdin(%rip), %rax
	movq	%rax, -24(%rbp)
	movq	stdout(%rip), %rax
	movq	%rax, -16(%rbp)
	movq	$4, -8(%rbp)
	jmp	.L44
.L30:
	movq	-16(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	disemvowel
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	fclose@PLT
	movq	$8, -8(%rbp)
	jmp	.L44
.L34:
	movl	$0, %eax
	jmp	.L45
.L40:
	movq	-48(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC0(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$3, -8(%rbp)
	jmp	.L44
.L38:
	movl	$1, %eax
	jmp	.L45
.L31:
	movq	$14, -8(%rbp)
	jmp	.L44
.L33:
	cmpq	$0, -16(%rbp)
	jne	.L46
	movq	$1, -8(%rbp)
	jmp	.L44
.L46:
	movq	$12, -8(%rbp)
	jmp	.L44
.L29:
	movl	$1, %eax
	jmp	.L45
.L36:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	leaq	.LC1(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -24(%rbp)
	movq	$0, -8(%rbp)
	jmp	.L44
.L32:
	movq	-48(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC0(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movq	$13, -8(%rbp)
	jmp	.L44
.L41:
	cmpq	$0, -24(%rbp)
	jne	.L48
	movq	$10, -8(%rbp)
	jmp	.L44
.L48:
	movq	$2, -8(%rbp)
	jmp	.L44
.L35:
	movq	-48(%rbp), %rax
	addq	$16, %rax
	movq	(%rax), %rax
	leaq	.LC2(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	fopen@PLT
	movq	%rax, -16(%rbp)
	movq	$9, -8(%rbp)
	jmp	.L44
.L39:
	cmpl	$2, -36(%rbp)
	jle	.L50
	movq	$7, -8(%rbp)
	jmp	.L44
.L50:
	movq	$12, -8(%rbp)
	jmp	.L44
.L53:
	nop
.L44:
	jmp	.L52
.L45:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
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
