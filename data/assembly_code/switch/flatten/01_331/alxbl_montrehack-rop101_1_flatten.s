	.file	"alxbl_montrehack-rop101_1_flatten.c"
	.text
	.globl	_TIG_IZ_pj3m_envp
	.bss
	.align 8
	.type	_TIG_IZ_pj3m_envp, @object
	.size	_TIG_IZ_pj3m_envp, 8
_TIG_IZ_pj3m_envp:
	.zero	8
	.globl	_TIG_IZ_pj3m_argv
	.align 8
	.type	_TIG_IZ_pj3m_argv, @object
	.size	_TIG_IZ_pj3m_argv, 8
_TIG_IZ_pj3m_argv:
	.zero	8
	.globl	motd
	.align 32
	.type	motd, @object
	.size	motd, 256
motd:
	.zero	256
	.globl	_TIG_IZ_pj3m_argc
	.align 4
	.type	_TIG_IZ_pj3m_argc, @object
	.size	_TIG_IZ_pj3m_argc, 4
_TIG_IZ_pj3m_argc:
	.zero	4
	.globl	done
	.align 4
	.type	done, @object
	.size	done, 4
done:
	.zero	4
	.section	.rodata
	.align 8
.LC0:
	.string	"Type in the new message of the day please:\n> "
	.text
	.globl	read_motd
	.type	read_motd, @function
read_motd:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$312, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -312(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	$9, -296(%rbp)
.L14:
	cmpq	$9, -296(%rbp)
	ja	.L17
	movq	-296(%rbp), %rax
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
	.long	.L17-.L4
	.long	.L18-.L4
	.long	.L17-.L4
	.long	.L17-.L4
	.long	.L17-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L5:
	movl	-300(%rbp), %eax
	movb	$0, -288(%rbp,%rax)
	addl	$1, -300(%rbp)
	movq	$5, -296(%rbp)
	jmp	.L10
.L3:
	movq	$7, -296(%rbp)
	jmp	.L10
.L7:
	leaq	-288(%rbp), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	gets@PLT
	movq	-312(%rbp), %rax
	movq	-288(%rbp), %rcx
	movq	-280(%rbp), %rbx
	movq	%rcx, (%rax)
	movq	%rbx, 8(%rax)
	movq	-272(%rbp), %rcx
	movq	-264(%rbp), %rbx
	movq	%rcx, 16(%rax)
	movq	%rbx, 24(%rax)
	movq	-256(%rbp), %rcx
	movq	-248(%rbp), %rbx
	movq	%rcx, 32(%rax)
	movq	%rbx, 40(%rax)
	movq	-240(%rbp), %rcx
	movq	-232(%rbp), %rbx
	movq	%rcx, 48(%rax)
	movq	%rbx, 56(%rax)
	movq	-224(%rbp), %rcx
	movq	-216(%rbp), %rbx
	movq	%rcx, 64(%rax)
	movq	%rbx, 72(%rax)
	movq	-208(%rbp), %rcx
	movq	-200(%rbp), %rbx
	movq	%rcx, 80(%rax)
	movq	%rbx, 88(%rax)
	movq	-192(%rbp), %rcx
	movq	-184(%rbp), %rbx
	movq	%rcx, 96(%rax)
	movq	%rbx, 104(%rax)
	movq	-176(%rbp), %rcx
	movq	-168(%rbp), %rbx
	movq	%rcx, 112(%rax)
	movq	%rbx, 120(%rax)
	movq	-160(%rbp), %rcx
	movq	-152(%rbp), %rbx
	movq	%rcx, 128(%rax)
	movq	%rbx, 136(%rax)
	movq	-144(%rbp), %rcx
	movq	-136(%rbp), %rbx
	movq	%rcx, 144(%rax)
	movq	%rbx, 152(%rax)
	movq	-128(%rbp), %rcx
	movq	-120(%rbp), %rbx
	movq	%rcx, 160(%rax)
	movq	%rbx, 168(%rax)
	movq	-112(%rbp), %rcx
	movq	-104(%rbp), %rbx
	movq	%rcx, 176(%rax)
	movq	%rbx, 184(%rax)
	movq	-96(%rbp), %rcx
	movq	-88(%rbp), %rbx
	movq	%rcx, 192(%rax)
	movq	%rbx, 200(%rax)
	movq	-80(%rbp), %rcx
	movq	-72(%rbp), %rbx
	movq	%rcx, 208(%rax)
	movq	%rbx, 216(%rax)
	movq	-64(%rbp), %rcx
	movq	-56(%rbp), %rbx
	movq	%rcx, 224(%rax)
	movq	%rbx, 232(%rax)
	movq	-48(%rbp), %rcx
	movq	-40(%rbp), %rbx
	movq	%rcx, 240(%rax)
	movq	%rbx, 248(%rax)
	movq	$1, -296(%rbp)
	jmp	.L10
.L8:
	cmpl	$255, -300(%rbp)
	jbe	.L12
	movq	$6, -296(%rbp)
	jmp	.L10
.L12:
	movq	$8, -296(%rbp)
	jmp	.L10
.L6:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movb	$0, -288(%rbp)
	movl	$1, -300(%rbp)
	movq	$5, -296(%rbp)
	jmp	.L10
.L17:
	nop
.L10:
	jmp	.L14
.L18:
	nop
	movq	-24(%rbp), %rax
	subq	%fs:40, %rax
	je	.L16
	call	__stack_chk_fail@PLT
.L16:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	read_motd, .-read_motd
	.section	.rodata
.LC1:
	.string	"%d"
	.text
	.globl	getnum
	.type	getnum, @function
getnum:
.LFB6:
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
	movq	$1, -16(%rbp)
.L25:
	cmpq	$2, -16(%rbp)
	je	.L20
	cmpq	$2, -16(%rbp)
	ja	.L28
	cmpq	$0, -16(%rbp)
	je	.L22
	cmpq	$1, -16(%rbp)
	jne	.L28
	movq	$2, -16(%rbp)
	jmp	.L23
.L22:
	movl	-20(%rbp), %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L26
	jmp	.L27
.L20:
	movl	$-1, -20(%rbp)
	leaq	-20(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	call	clean_stdin
	movq	$0, -16(%rbp)
	jmp	.L23
.L28:
	nop
.L23:
	jmp	.L25
.L27:
	call	__stack_chk_fail@PLT
.L26:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	getnum, .-getnum
	.section	.rodata
	.align 8
.LC2:
	.string	"motd daemon v0.1 (c) 2019 BetterSoft"
.LC3:
	.string	"date"
.LC4:
	.string	"=> How may I help you today?"
	.align 8
.LC5:
	.string	"    1 - View message of the day"
	.align 8
.LC6:
	.string	"    2 - Change message of the day"
.LC7:
	.string	"    3 - Exit"
.LC8:
	.string	"> "
	.align 8
.LC9:
	.string	"I don't recognize that option."
.LC10:
	.string	"Bye!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB8:
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
	movl	$0, done(%rip)
	nop
.L30:
	movb	$62, motd(%rip)
	movb	$32, 1+motd(%rip)
	movb	$62, 2+motd(%rip)
	movb	$32, 3+motd(%rip)
	movb	$62, 4+motd(%rip)
	movb	$32, 5+motd(%rip)
	movb	$68, 6+motd(%rip)
	movb	$45, 7+motd(%rip)
	movb	$100, 8+motd(%rip)
	movb	$45, 9+motd(%rip)
	movb	$100, 10+motd(%rip)
	movb	$45, 11+motd(%rip)
	movb	$68, 12+motd(%rip)
	movb	$82, 13+motd(%rip)
	movb	$79, 14+motd(%rip)
	movb	$80, 15+motd(%rip)
	movb	$32, 16+motd(%rip)
	movb	$116, 17+motd(%rip)
	movb	$104, 18+motd(%rip)
	movb	$101, 19+motd(%rip)
	movb	$32, 20+motd(%rip)
	movb	$82, 21+motd(%rip)
	movb	$79, 22+motd(%rip)
	movb	$80, 23+motd(%rip)
	movb	$33, 24+motd(%rip)
	movb	$32, 25+motd(%rip)
	movb	$60, 26+motd(%rip)
	movb	$32, 27+motd(%rip)
	movb	$60, 28+motd(%rip)
	movb	$32, 29+motd(%rip)
	movb	$60, 30+motd(%rip)
	movb	$0, 31+motd(%rip)
	nop
.L31:
	movq	$0, _TIG_IZ_pj3m_envp(%rip)
	nop
.L32:
	movq	$0, _TIG_IZ_pj3m_argv(%rip)
	nop
.L33:
	movl	$0, _TIG_IZ_pj3m_argc(%rip)
	nop
	nop
.L34:
.L35:
#APP
# 139 "alxbl_montrehack-rop101_1.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-pj3m--0
# 0 "" 2
#NO_APP
	movl	-20(%rbp), %eax
	movl	%eax, _TIG_IZ_pj3m_argc(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, _TIG_IZ_pj3m_argv(%rip)
	movq	-40(%rbp), %rax
	movq	%rax, _TIG_IZ_pj3m_envp(%rip)
	nop
	movq	$6, -8(%rbp)
.L58:
	cmpq	$15, -8(%rbp)
	ja	.L59
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L38(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L38(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L38:
	.long	.L48-.L38
	.long	.L47-.L38
	.long	.L46-.L38
	.long	.L45-.L38
	.long	.L59-.L38
	.long	.L59-.L38
	.long	.L44-.L38
	.long	.L43-.L38
	.long	.L59-.L38
	.long	.L59-.L38
	.long	.L42-.L38
	.long	.L59-.L38
	.long	.L41-.L38
	.long	.L40-.L38
	.long	.L39-.L38
	.long	.L60-.L38
	.text
.L39:
	leaq	motd(%rip), %rax
	movq	%rax, %rdi
	call	read_motd
	movq	$12, -8(%rbp)
	jmp	.L49
.L41:
	movl	done(%rip), %eax
	testl	%eax, %eax
	jne	.L51
	movq	$10, -8(%rbp)
	jmp	.L49
.L51:
	movq	$2, -8(%rbp)
	jmp	.L49
.L47:
	movq	stdout(%rip), %rax
	movl	$0, %ecx
	movl	$2, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	setvbuf@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	stdout(%rip), %rax
	movq	%rax, %rdi
	call	fflush@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	system@PLT
	movq	$12, -8(%rbp)
	jmp	.L49
.L45:
	movl	$1, done(%rip)
	movq	$12, -8(%rbp)
	jmp	.L49
.L40:
	cmpl	$3, -16(%rbp)
	je	.L53
	cmpl	$3, -16(%rbp)
	jg	.L54
	cmpl	$1, -16(%rbp)
	je	.L55
	cmpl	$2, -16(%rbp)
	je	.L56
	jmp	.L54
.L53:
	movq	$3, -8(%rbp)
	jmp	.L57
.L56:
	movq	$14, -8(%rbp)
	jmp	.L57
.L55:
	movq	$7, -8(%rbp)
	jmp	.L57
.L54:
	movq	$0, -8(%rbp)
	nop
.L57:
	jmp	.L49
.L44:
	movq	$1, -8(%rbp)
	jmp	.L49
.L42:
	movl	$10, %edi
	call	putchar@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	call	getnum
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movl	%eax, -16(%rbp)
	movl	$10, %edi
	call	putchar@PLT
	movq	$13, -8(%rbp)
	jmp	.L49
.L48:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -8(%rbp)
	jmp	.L49
.L43:
	leaq	motd(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$12, -8(%rbp)
	jmp	.L49
.L46:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$15, -8(%rbp)
	jmp	.L49
.L59:
	nop
.L49:
	jmp	.L58
.L60:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	main, .-main
	.globl	clean_stdin
	.type	clean_stdin, @function
clean_stdin:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$7, -8(%rbp)
.L76:
	cmpq	$7, -8(%rbp)
	ja	.L77
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L64(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L64(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L64:
	.long	.L69-.L64
	.long	.L68-.L64
	.long	.L67-.L64
	.long	.L77-.L64
	.long	.L78-.L64
	.long	.L77-.L64
	.long	.L65-.L64
	.long	.L63-.L64
	.text
.L68:
	cmpb	$10, -13(%rbp)
	jne	.L71
	movq	$4, -8(%rbp)
	jmp	.L73
.L71:
	movq	$6, -8(%rbp)
	jmp	.L73
.L65:
	cmpb	$-1, -13(%rbp)
	jne	.L74
	movq	$2, -8(%rbp)
	jmp	.L73
.L74:
	movq	$0, -8(%rbp)
	jmp	.L73
.L69:
	call	getchar@PLT
	movl	%eax, -12(%rbp)
	movl	-12(%rbp), %eax
	movb	%al, -13(%rbp)
	movq	$1, -8(%rbp)
	jmp	.L73
.L63:
	movq	$0, -8(%rbp)
	jmp	.L73
.L67:
	movl	$1, %edi
	call	exit@PLT
.L77:
	nop
.L73:
	jmp	.L76
.L78:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE9:
	.size	clean_stdin, .-clean_stdin
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
